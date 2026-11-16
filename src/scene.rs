//! State shared between the ROS thread, the image decoders and the render loop. Writers replace whole values under a
//! short lock (latest message wins, nothing queues up); the render loop compares versions and uploads only what changed.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::{DAffine3, DQuat, DVec3};

use crate::msgs::CloudColor;

pub struct CloudData {
    pub frame_id: String,
    pub n: u32,
    pub point_step: u32,
    pub xyz: [u32; 3],
    pub color: CloudColor,
    pub range: (f32, f32),
    /// The received message (or the repacked points); points are `bytes[off..off + len]`.
    pub bytes: Arc<Vec<u8>>,
    pub off: usize,
    pub len: usize,
    /// Spatial chunks of a big cloud (empty for a small one).
    pub chunks: Arc<Vec<crate::msgs::Chunk>>,
    pub version: u64,
}

pub struct ImageData {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, tightly packed.
    pub rgba: Arc<Vec<u8>>,
    pub version: u64,
    pub decode_ms: f32,
}

pub struct LineData {
    pub frame_id: String,
    pub points: VecDeque<[f64; 3]>,
    pub version: u64,
}

#[derive(Default)]
pub struct TopicStat {
    pub kind: &'static str,
    pub qos: String,
    pub count: u64,
    pub bytes_last: usize,
    pub window: VecDeque<Instant>,
    pub error: Option<String>,
}

impl TopicStat {
    pub fn hz(&self) -> f32 {
        let now = Instant::now();
        let n = self.window.iter().filter(|t| now.duration_since(**t).as_secs_f32() < 2.0).count();
        n as f32 / 2.0
    }
    pub fn tick(&mut self, bytes: usize) {
        self.count += 1;
        self.bytes_last = bytes;
        let now = Instant::now();
        self.window.push_back(now);
        while self.window.front().is_some_and(|t| now.duration_since(*t).as_secs_f32() > 2.0) {
            self.window.pop_front();
        }
    }
}

#[derive(Default)]
pub struct TfTree {
    /// child -> (parent, transform parent <- child)
    edges: HashMap<String, (String, DAffine3)>,
    /// Child frames whose transform came from /tf_static.
    statics: std::collections::HashSet<String>,
    pub version: u64,
}

impl TfTree {
    pub fn set(&mut self, parent: &str, child: &str, t: [f64; 3], q: [f64; 4]) {
        let a = DAffine3::from_rotation_translation(DQuat::from_xyzw(q[0], q[1], q[2], q[3]).normalize(), DVec3::from(t));
        self.edges.insert(child.trim_start_matches('/').to_owned(), (parent.trim_start_matches('/').to_owned(), a));
        self.version += 1;
    }

    /// root <- frame, and the root's name.
    fn to_root(&self, frame: &str) -> (String, DAffine3) {
        let mut f = frame.trim_start_matches('/').to_owned();
        let mut acc = DAffine3::IDENTITY;
        for _ in 0..64 {
            match self.edges.get(&f) {
                Some((p, a)) => {
                    acc = *a * acc;
                    f = p.clone();
                }
                None => break,
            }
        }
        (f, acc)
    }

    /// fixed <- frame with the latest transforms, or None when the two frames are not connected.
    pub fn lookup(&self, fixed: &str, frame: &str) -> Option<DAffine3> {
        if frame.is_empty() || frame.trim_start_matches('/') == fixed.trim_start_matches('/') {
            return Some(DAffine3::IDENTITY);
        }
        let (r1, a1) = self.to_root(frame);
        let (r2, a2) = self.to_root(fixed);
        (r1 == r2).then(|| a2.inverse() * a1)
    }

    /// The frame every other hangs from: the one that is never a child (with several trees, the one with most
    /// frames under it).
    pub fn root(&self) -> Option<String> {
        let mut size: HashMap<String, usize> = HashMap::new();
        for child in self.edges.keys() {
            *size.entry(self.to_root(child).0).or_default() += 1;
        }
        size.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map(|(f, _)| f)
    }

    /// The transform of this child frame is a static one (/tf_static).
    pub fn mark_static(&mut self, child: &str) {
        self.statics.insert(child.trim_start_matches('/').to_owned());
    }

    /// The frames set by /tf_static, sorted.
    pub fn static_frames(&self) -> Vec<String> {
        let mut v: Vec<String> = self.statics.iter().filter(|f| self.edges.contains_key(*f)).cloned().collect();
        v.sort();
        v
    }

    pub fn clear(&mut self) {
        self.edges.clear();
        self.statics.clear();
        self.version += 1;
    }

    pub fn frames(&self) -> Vec<String> {
        let mut v: Vec<String> = self.edges.keys().cloned().chain(self.edges.values().map(|e| e.0.clone())).collect();
        v.sort();
        v.dedup();
        v
    }
}

/// Vertex of every triangle mesh (markers, robot model): position in the object's frame, linear-free sRGB colour, UV.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

/// A decoded RGBA8 texture (marker TRIANGLE_LIST textures).
pub struct Texture2D {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub enum Prim {
    /// Segment pairs (LINE_LIST layout) with one colour per vertex.
    Lines(Vec<[f32; 3]>, Vec<[f32; 4]>),
    Mesh { verts: Arc<Vec<MeshVertex>>, texture: Option<Arc<Texture2D>>, shade: bool },
    Text { pos: [f32; 3], text: String, color: [f32; 4] },
}

pub struct MarkerData {
    pub frame_id: String,
    /// frame <- marker
    pub pose: DAffine3,
    pub prim: Prim,
    pub expires: Option<Instant>,
    pub version: u64,
}

pub struct RobotVisual {
    pub link: String,
    /// link <- visual
    pub origin: DAffine3,
    pub verts: Arc<Vec<MeshVertex>>,
}

pub struct Robot {
    pub visuals: Vec<RobotVisual>,
    pub version: u64,
}

#[derive(Default)]
pub struct SceneData {
    pub clouds: HashMap<String, CloudData>,
    pub images: HashMap<String, ImageData>,
    pub lines: HashMap<String, LineData>,
    /// topic -> (namespace, id) -> marker
    pub markers: HashMap<String, HashMap<(String, i32), MarkerData>>,
    pub robots: HashMap<String, Robot>,
    pub tf: TfTree,
    pub stats: HashMap<String, TopicStat>,
    pub version: u64,
}

/// Wakes the render loop at most once per frame, whatever the message rate.
pub struct Waker {
    pending: AtomicBool,
    proxy: winit::event_loop::EventLoopProxy<()>,
}

impl Waker {
    pub fn new(proxy: winit::event_loop::EventLoopProxy<()>) -> Self {
        Self { pending: AtomicBool::new(false), proxy }
    }
    pub fn wake(&self) {
        if !self.pending.swap(true, Ordering::AcqRel) {
            let _ = self.proxy.send_event(());
        }
    }
    pub fn clear(&self) {
        self.pending.store(false, Ordering::Release);
    }
}

/// What the top bar reports about bag loading.
#[derive(Default)]
pub struct Status {
    /// "Opening x.mcap" while a bag is being indexed.
    pub loading: Option<String>,
    /// Last error or warning, until dismissed.
    pub notice: Option<String>,
}

pub struct Shared {
    pub scene: Mutex<SceneData>,
    pub waker: Waker,
    /// False while a bag is open: the live ROS side is muted.
    pub live: AtomicBool,
    /// ROS domain the live side listens on (r2r has one rcl context per process: another domain is a restart).
    pub domain: std::sync::atomic::AtomicU32,
    /// The open bag, if any.
    pub player: Mutex<Option<Arc<crate::player::Playback>>>,
    /// The thread that plays it (or is still opening it) and its cancel flag.
    pub player_thread: Mutex<Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)>>,
    pub status: Mutex<Status>,
}

impl Shared {
    pub fn new(proxy: winit::event_loop::EventLoopProxy<()>) -> Self {
        Self {
            scene: Mutex::new(SceneData::default()),
            waker: Waker::new(proxy),
            live: AtomicBool::new(true),
            domain: std::sync::atomic::AtomicU32::new(std::env::var("ROS_DOMAIN_ID").ok().and_then(|d| d.trim().parse().ok()).unwrap_or(0)),
            player: Mutex::new(None),
            player_thread: Mutex::new(None),
            status: Mutex::new(Status::default()),
        }
    }
}
