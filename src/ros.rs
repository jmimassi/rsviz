//! ROS 2 side: one rcl node (r2r) on its own thread. Topics of the supported types are discovered every second and
//! subscribed as raw serialized messages (no conversion to Rust structs), with the publishers' reliability/durability
//! (a BEST_EFFORT camera stays best effort, a RELIABLE cloud is read reliably, latched /tf_static is transient local).
//! Small messages are decoded inline; images are handed to one decoder thread per topic through a latest-only slot.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use futures::executor::LocalPool;
use futures::task::LocalSpawnExt;
use futures::StreamExt;
use r2r::qos::{DurabilityPolicy, HistoryPolicy, ReliabilityPolicy};
use r2r::QosProfile;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;

use crate::msgs;
use crate::scene::{CloudData, ImageData, LineData, MarkerData, Prim, Shared};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Cloud,
    Image,
    Compressed,
    Path,
    Odom,
    Pose,
    Tf,
    Marker,
    MarkerArray,
    Point,
    RobotDesc,
}

impl Kind {
    pub fn of(type_name: &str) -> Option<Kind> {
        Some(match type_name {
            "sensor_msgs/msg/PointCloud2" => Kind::Cloud,
            "sensor_msgs/msg/Image" => Kind::Image,
            "sensor_msgs/msg/CompressedImage" => Kind::Compressed,
            "nav_msgs/msg/Path" => Kind::Path,
            "nav_msgs/msg/Odometry" => Kind::Odom,
            // Same leading layout (header, pose); the covariance that follows is not read.
            "geometry_msgs/msg/PoseStamped" | "geometry_msgs/msg/PoseWithCovarianceStamped" => Kind::Pose,
            "tf2_msgs/msg/TFMessage" => Kind::Tf,
            "visualization_msgs/msg/Marker" => Kind::Marker,
            "visualization_msgs/msg/MarkerArray" => Kind::MarkerArray,
            "geometry_msgs/msg/PointStamped" => Kind::Point,
            _ => return None,
        })
    }
    /// Kind of a topic from its name and type (a std_msgs/String is a robot model only on a robot_description topic).
    pub fn of_topic(name: &str, type_name: &str) -> Option<Kind> {
        Kind::of(type_name).or_else(|| (type_name == "std_msgs/msg/String" && name.ends_with("robot_description")).then_some(Kind::RobotDesc))
    }
    /// Topics whose last message alone is the state (the others accumulate: transforms, trails, markers).
    pub fn latest_only(self) -> bool {
        matches!(self, Kind::Cloud | Kind::Image | Kind::Compressed | Kind::Path | Kind::Point)
    }
    pub fn label(self) -> &'static str {
        match self {
            Kind::Cloud => "PointCloud2",
            Kind::Image => "Image",
            Kind::Compressed => "CompressedImage",
            Kind::Path => "Path",
            Kind::Odom => "Odometry",
            Kind::Pose => "PoseStamped",
            Kind::Tf => "TF",
            Kind::Marker => "Marker",
            Kind::MarkerArray => "MarkerArray",
            Kind::Point => "PointStamped",
            Kind::RobotDesc => "RobotModel",
        }
    }
}

pub struct Options {
    /// Only these topics (all supported topics when None).
    pub topics: Option<Vec<String>>,
    /// Max positions kept per Odometry / PoseStamped trail.
    pub trail: usize,
}

pub fn spawn(shared: Arc<Shared>, opts: Options) {
    std::thread::Builder::new()
        .name("ros".into())
        .spawn(move || {
            log::info!("listening on ROS domain {}", shared.domain.load(Ordering::Acquire));
            if let Err(e) = run(shared, opts) {
                log::error!("ROS thread stopped: {e}");
            }
        })
        .expect("spawn ROS thread");
}

fn matched_qos(pubs: &[QosProfile], kind: Kind) -> (QosProfile, String) {
    let best_effort = pubs.iter().any(|p| p.reliability == ReliabilityPolicy::BestEffort);
    let latched = !pubs.is_empty() && pubs.iter().all(|p| p.durability == DurabilityPolicy::TransientLocal);
    let mut q = QosProfile::default();
    q.history = HistoryPolicy::KeepLast;
    q.reliability = if best_effort { ReliabilityPolicy::BestEffort } else { ReliabilityPolicy::Reliable };
    q.durability = if latched { DurabilityPolicy::TransientLocal } else { DurabilityPolicy::Volatile };
    // Heavy messages: a short reader history, the render loop only ever wants the latest one.
    q.depth = match kind {
        Kind::Tf => 100,
        Kind::Path | Kind::Odom | Kind::Pose | Kind::Point => 50,
        Kind::Marker | Kind::MarkerArray => 20,
        _ => 2,
    };
    let desc = format!("{}{}", if best_effort { "best effort" } else { "reliable" }, if latched { ", latched" } else { "" });
    (q, desc)
}

fn run(shared: Arc<Shared>, opts: Options) -> r2r::Result<()> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "rsviz", "")?;
    let mut pool = LocalPool::new();
    let spawner = pool.spawner();
    let mut subscribed: HashSet<String> = HashSet::new();
    let mut last_discovery = Instant::now() - Duration::from_secs(10);
    let opts = Arc::new(opts);
    let mut handlers: Vec<Arc<Handler>> = Vec::new();
    let mut was_live = true;

    loop {
        // While a bag is open the live side is muted: nothing new is subscribed and received messages are dropped
        // (latched ones are kept and given back when the bag is closed).
        let live = shared.live.load(Ordering::Acquire);
        if live && !was_live {
            for h in &handlers {
                h.resume();
            }
        }
        was_live = live;
        if live && last_discovery.elapsed() >= Duration::from_secs(1) {
            last_discovery = Instant::now();
            for (name, types) in node.get_topic_names_and_types().unwrap_or_default() {
                if subscribed.contains(&name) || opts.topics.as_ref().is_some_and(|f| !f.contains(&name)) {
                    continue;
                }
                let Some((type_name, kind)) = types.iter().find_map(|t| Kind::of_topic(&name, t).map(|k| (t.clone(), k))) else { continue };
                let pubs: Vec<QosProfile> = node.get_publishers_info_by_topic(&name, false).unwrap_or_default().into_iter().map(|p| p.qos_profile).collect();
                if pubs.is_empty() {
                    continue; // wait for a publisher, to copy its QoS
                }
                let (qos, desc) = matched_qos(&pubs, kind);
                match node.subscribe_raw(&name, &type_name, qos) {
                    Ok(mut stream) => {
                        log::info!("subscribed {name} ({}, {desc})", kind.label());
                        subscribed.insert(name.clone());
                        let h = Arc::new(Handler::new(shared.clone(), name.clone(), kind, opts.trail, desc, false));
                        handlers.push(h.clone());
                        let _ = spawner.spawn_local(async move {
                            while let Some(msg) = stream.next().await {
                                h.handle(msg);
                            }
                        });
                    }
                    Err(e) => log::warn!("cannot subscribe {name}: {e}"),
                }
            }
        }
        // Blocks on the rcl wait set until a message is ready (or 50 ms), then runs the subscription tasks. With no
        // subscription yet the wait set is empty and rcl returns at once: sleep instead of spinning a core.
        if subscribed.is_empty() {
            std::thread::sleep(Duration::from_millis(100));
        } else {
            node.spin_once(Duration::from_millis(50));
        }
        pool.run_until_stalled();
    }
}

/// Latest-only hand-off to a decoder thread: a new message replaces one that was not decoded yet.
struct Slot {
    msg: Mutex<Option<Vec<u8>>>,
    cv: Condvar,
    /// Set when the handler is dropped: the decoder thread ends and publishes nothing more.
    closed: AtomicBool,
}

/// Turns the serialized messages of one topic into scene content. Fed by the live subscriptions and by the bag player.
pub struct Handler {
    shared: Arc<Shared>,
    topic: String,
    kind: Kind,
    trail: usize,
    /// Shown next to the topic ("reliable", "bag" ...).
    desc: String,
    /// Messages come from a bag (a live handler is muted while a bag is open).
    bag: bool,
    slot: Option<Arc<Slot>>,
    /// Live latched messages (static transforms, robot model), to restore the scene after a bag is closed.
    latched: Mutex<Vec<Vec<u8>>>,
}

impl Drop for Handler {
    fn drop(&mut self) {
        if let Some(slot) = &self.slot {
            slot.closed.store(true, Ordering::Release);
            slot.cv.notify_all();
        }
    }
}

impl Handler {
    pub fn new(shared: Arc<Shared>, topic: String, kind: Kind, trail: usize, desc: String, bag: bool) -> Self {
        let slot = matches!(kind, Kind::Image | Kind::Compressed).then(|| {
            let slot = Arc::new(Slot { msg: Mutex::new(None), cv: Condvar::new(), closed: AtomicBool::new(false) });
            let (s2, sh, t) = (slot.clone(), shared.clone(), topic.clone());
            std::thread::Builder::new()
                .name(format!("img {topic}"))
                .spawn(move || image_worker(s2, sh, t, kind, bag))
                .expect("spawn image decoder");
            slot
        });
        let h = Self { shared, topic, kind, trail, desc, bag, slot, latched: Mutex::new(Vec::new()) };
        h.announce();
        h
    }

    /// Lists the topic in the scene statistics (kind and source).
    fn announce(&self) {
        let mut s = self.shared.scene.lock().unwrap();
        let st = s.stats.entry(self.topic.clone()).or_default();
        st.kind = self.kind.label();
        st.qos = self.desc.clone();
    }

    fn is_latched(&self) -> bool {
        self.kind == Kind::RobotDesc || (self.kind == Kind::Tf && self.topic.ends_with("tf_static"))
    }

    /// Live handler, when a bag has just been closed: the topic is listed again and its latched messages re-applied.
    fn resume(&self) {
        self.announce();
        let msgs = self.latched.lock().unwrap().clone();
        for m in msgs {
            self.process(m);
        }
    }

    pub fn handle(&self, msg: Vec<u8>) {
        if !self.bag {
            if self.is_latched() {
                let mut l = self.latched.lock().unwrap();
                if self.kind == Kind::RobotDesc {
                    l.clear();
                }
                if l.len() < 256 {
                    l.push(msg.clone());
                }
            }
            if !self.shared.live.load(Ordering::Acquire) {
                return;
            }
        }
        self.process(msg);
    }

    fn process(&self, msg: Vec<u8>) {
        let len = msg.len();
        if let Some(slot) = &self.slot {
            *slot.msg.lock().unwrap() = Some(msg);
            slot.cv.notify_one();
            self.shared.scene.lock().unwrap().stats.entry(self.topic.clone()).or_default().tick(len);
            return;
        }
        let res = self.decode_small(msg);
        let mut s = self.shared.scene.lock().unwrap();
        let st = s.stats.entry(self.topic.clone()).or_default();
        st.tick(len);
        if let Err(e) = res {
            st.error = Some(e.0.to_string());
        }
        drop(s);
        self.shared.waker.wake();
    }

    fn decode_small(&self, msg: Vec<u8>) -> Result<(), crate::cdr::CdrError> {
        match self.kind {
            Kind::Cloud => {
                let (c, repacked) = msgs::point_cloud2(&msg)?;
                let bytes = Arc::new(repacked.unwrap_or(msg));
                let chunks = Arc::new(msgs::cloud_chunks(&bytes[c.data_off..c.data_off + c.data_len], c.n, c.point_step, c.xyz));
                let mut s = self.shared.scene.lock().unwrap();
                s.version += 1;
                let version = s.version;
                s.clouds.insert(
                    self.topic.clone(),
                    CloudData {
                        frame_id: c.header.frame_id,
                        n: c.n,
                        point_step: c.point_step,
                        xyz: c.xyz,
                        color: c.color,
                        range: c.range,
                        bytes,
                        off: c.data_off,
                        len: c.data_len,
                        chunks,
                        version,
                    },
                );
            }
            Kind::Path => {
                let (h, pts) = msgs::path(&msg)?;
                let mut s = self.shared.scene.lock().unwrap();
                s.version += 1;
                let version = s.version;
                s.lines.insert(self.topic.clone(), LineData { frame_id: h.frame_id, points: pts.into(), version });
            }
            Kind::Odom | Kind::Pose => {
                let (h, p, _q) = if self.kind == Kind::Odom { msgs::odometry(&msg)? } else { msgs::pose_stamped(&msg)? };
                let mut s = self.shared.scene.lock().unwrap();
                s.version += 1;
                let version = s.version;
                let l = s.lines.entry(self.topic.clone()).or_insert_with(|| LineData { frame_id: String::new(), points: Default::default(), version });
                l.frame_id = h.frame_id;
                l.points.push_back(p);
                while l.points.len() > self.trail {
                    l.points.pop_front();
                }
                l.version = version;
            }
            Kind::Tf => {
                let tfs = msgs::tf_message(&msg)?;
                let mut s = self.shared.scene.lock().unwrap();
                for t in &tfs {
                    s.tf.set(&t.parent, &t.child, t.t, t.q);
                    if self.topic.ends_with("tf_static") {
                        s.tf.mark_static(&t.child);
                    }
                }
                s.version += 1;
            }
            Kind::Marker | Kind::MarkerArray => {
                let ms = if self.kind == Kind::Marker { msgs::marker_msg(&msg)? } else { msgs::marker_array(&msg)? };
                // Geometry (and texture decoding) outside the lock.
                let built: Vec<_> = ms.iter().map(|m| (m.action, m.ns.clone(), m.id, (m.action != 2 && m.action != 3).then(|| crate::shapes::marker_prim(m)), m)).collect();
                let now = Instant::now();
                let mut err = None;
                let mut s = self.shared.scene.lock().unwrap();
                s.version += 1;
                let version = s.version;
                let map = s.markers.entry(self.topic.clone()).or_default();
                for (action, ns, id, prim, m) in built {
                    match (action, prim) {
                        (3, _) => map.retain(|k, _| !ns.is_empty() && k.0 != ns),
                        (2, _) => {
                            map.remove(&(ns, id));
                        }
                        (_, Some(Ok(prim))) => {
                            let expires = (m.lifetime > 0.0).then(|| now + Duration::from_secs_f64(m.lifetime));
                            map.insert((ns, id), MarkerData { frame_id: m.frame_id.clone(), pose: crate::shapes::marker_pose(m), prim, expires, version });
                        }
                        (_, Some(Err(e))) => err = Some(e),
                        _ => {}
                    }
                }
                if let Some(e) = err {
                    s.stats.entry(self.topic.clone()).or_default().error = Some(e);
                }
            }
            Kind::Point => {
                let (h, p) = msgs::point_stamped(&msg)?;
                let mut verts = Vec::new();
                crate::shapes::sphere(&mut verts, glam::Vec3::ZERO, glam::Vec3::splat(1.0), [1.0, 0.55, 0.1, 1.0]);
                let mut s = self.shared.scene.lock().unwrap();
                s.version += 1;
                let version = s.version;
                let prim = Prim::Mesh { verts: Arc::new(verts), texture: None, shade: true };
                let pose = glam::DAffine3::from_translation(glam::DVec3::from(p));
                s.markers.entry(self.topic.clone()).or_default().insert((String::new(), 0), MarkerData { frame_id: h.frame_id, pose, prim, expires: None, version });
            }
            Kind::RobotDesc => {
                let text = msgs::string(&msg)?;
                let (visuals, missing) = crate::urdf::parse(&text).map_err(|_| crate::cdr::CdrError("robot_description is not a valid URDF"))?;
                log::info!("{}: robot model with {} visuals", self.topic, visuals.len());
                let mut s = self.shared.scene.lock().unwrap();
                s.version += 1;
                let version = s.version;
                s.robots.insert(self.topic.clone(), crate::scene::Robot { visuals, version });
                if !missing.is_empty() {
                    s.stats.entry(self.topic.clone()).or_default().error = Some(format!("meshes not found: {}", missing.join(", ")));
                }
                drop(s);
                return Ok(());
            }
            Kind::Image | Kind::Compressed => unreachable!(),
        }
        Ok(())
    }
}

fn image_worker(slot: Arc<Slot>, shared: Arc<Shared>, topic: String, kind: Kind, bag: bool) {
    loop {
        let msg = {
            let mut g = slot.msg.lock().unwrap();
            while g.is_none() {
                if slot.closed.load(Ordering::Acquire) {
                    return;
                }
                g = slot.cv.wait(g).unwrap();
            }
            g.take().unwrap()
        };
        let t0 = Instant::now();
        let res = if kind == Kind::Compressed { decode_compressed(&msg) } else { decode_raw(&msg) };
        let ms = t0.elapsed().as_secs_f32() * 1e3;
        let mut s = shared.scene.lock().unwrap();
        // Checked under the scene lock: nothing is published once the source of this image was closed or muted.
        if slot.closed.load(Ordering::Acquire) || (!bag && !shared.live.load(Ordering::Acquire)) {
            continue;
        }
        match res {
            Ok((width, height, rgba)) => {
                s.version += 1;
                let version = s.version;
                s.images.insert(topic.clone(), ImageData { width, height, rgba: Arc::new(rgba), version, decode_ms: ms });
                s.stats.entry(topic.clone()).or_default().error = None;
            }
            Err(e) => s.stats.entry(topic.clone()).or_default().error = Some(e),
        }
        drop(s);
        shared.waker.wake();
    }
}

fn decode_compressed(msg: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let c = msgs::compressed_image(msg).map_err(|e| e.0.to_string())?;
    let data = &msg[c.data_off..c.data_off + c.data_len];
    if c.format.to_lowercase().contains("png") || data.starts_with(&[0x89, b'P', b'N', b'G']) {
        return Err(format!("PNG compressed images are not supported ({})", c.format));
    }
    let mut d = zune_jpeg::JpegDecoder::new_with_options(data, DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA));
    let px = d.decode().map_err(|e| format!("JPEG: {e:?}"))?;
    let info = d.info().ok_or("JPEG without header")?;
    let rgba = crate::shapes::to_rgba(px, info.width as usize, info.height as usize).ok_or("JPEG with an unexpected number of channels")?;
    Ok((info.width as u32, info.height as u32, rgba))
}

fn decode_raw(msg: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let im = msgs::image(msg).map_err(|e| e.0.to_string())?;
    let (w, h, step) = (im.width as usize, im.height as usize, im.step as usize);
    let data = &msg[im.data_off..im.data_off + im.data_len];
    if data.len() < h * step {
        return Err("image data shorter than height * step".into());
    }
    let mut out = vec![255u8; w * h * 4];
    let enc = im.encoding.to_lowercase();
    let rd16 = |b: &[u8]| if im.big_endian { u16::from_be_bytes([b[0], b[1]]) } else { u16::from_le_bytes([b[0], b[1]]) };
    match enc.as_str() {
        "rgb8" | "bgr8" | "rgba8" | "bgra8" | "mono8" | "8uc1" | "8uc3" => {
            let ch = match enc.as_str() { "rgba8" | "bgra8" => 4, "mono8" | "8uc1" => 1, _ => 3 };
            let swap = enc.starts_with("bgr") || enc == "8uc3";
            for y in 0..h {
                let row = &data[y * step..y * step + w * ch];
                let o = &mut out[y * w * 4..(y + 1) * w * 4];
                for x in 0..w {
                    let p = &row[x * ch..x * ch + ch];
                    let q = &mut o[x * 4..x * 4 + 4];
                    if ch == 1 {
                        q[0] = p[0];
                        q[1] = p[0];
                        q[2] = p[0];
                    } else if swap {
                        q[0] = p[2];
                        q[1] = p[1];
                        q[2] = p[0];
                    } else {
                        q[..3].copy_from_slice(&p[..3]);
                    }
                }
            }
        }
        "mono16" | "16uc1" => {
            // Depth or 16-bit mono: normalized on the frame's own range.
            let (mut lo, mut hi) = (u16::MAX, 0u16);
            for y in 0..h {
                for x in 0..w {
                    let v = rd16(&data[y * step + 2 * x..]);
                    if v > 0 {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
            let span = (hi.saturating_sub(lo)).max(1) as f32;
            for y in 0..h {
                for x in 0..w {
                    let v = rd16(&data[y * step + 2 * x..]);
                    let g = if v == 0 { 0 } else { (((v - lo) as f32 / span) * 255.0) as u8 };
                    out[(y * w + x) * 4..(y * w + x) * 4 + 3].fill(g);
                }
            }
        }
        "32fc1" => {
            let rd = |b: &[u8]| { let a = [b[0], b[1], b[2], b[3]]; if im.big_endian { f32::from_be_bytes(a) } else { f32::from_le_bytes(a) } };
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for y in 0..h {
                for x in 0..w {
                    let v = rd(&data[y * step + 4 * x..]);
                    if v.is_finite() {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
            let span = (hi - lo).max(1e-6);
            for y in 0..h {
                for x in 0..w {
                    let v = rd(&data[y * step + 4 * x..]);
                    let g = if v.is_finite() { (((v - lo) / span) * 255.0) as u8 } else { 0 };
                    out[(y * w + x) * 4..(y * w + x) * 4 + 3].fill(g);
                }
            }
        }
        _ => return Err(format!("unsupported encoding {}", im.encoding)),
    }
    Ok((im.width, im.height, out))
}

/// One-line description of a decoded message (`rsviz --bag-info`): runs the same decoders as the viewer.
pub fn describe(kind: Kind, msg: &[u8]) -> Result<String, String> {
    let e = |e: crate::cdr::CdrError| e.0.to_string();
    Ok(match kind {
        Kind::Cloud => {
            let (c, repacked) = msgs::point_cloud2(msg).map_err(e)?;
            format!("{} points, frame {}{}", c.n, c.header.frame_id, if repacked.is_some() { ", repacked" } else { "" })
        }
        Kind::Image => {
            let im = msgs::image(msg).map_err(e)?;
            let (w, h, _) = decode_raw(msg)?;
            format!("{w}x{h} {}", im.encoding)
        }
        Kind::Compressed => {
            let c = msgs::compressed_image(msg).map_err(e)?;
            let (w, h, _) = decode_compressed(msg)?;
            format!("{w}x{h} {}", c.format)
        }
        Kind::Path => {
            let (h, pts) = msgs::path(msg).map_err(e)?;
            format!("{} poses, frame {}", pts.len(), h.frame_id)
        }
        Kind::Odom | Kind::Pose => {
            let (h, p, _) = if kind == Kind::Odom { msgs::odometry(msg) } else { msgs::pose_stamped(msg) }.map_err(e)?;
            format!("[{:.2} {:.2} {:.2}], frame {}", p[0], p[1], p[2], h.frame_id)
        }
        Kind::Tf => {
            let tfs = msgs::tf_message(msg).map_err(e)?;
            format!("{} transforms{}", tfs.len(), tfs.first().map(|t| format!(", {} -> {}", t.parent, t.child)).unwrap_or_default())
        }
        Kind::Marker | Kind::MarkerArray => {
            let ms = if kind == Kind::Marker { msgs::marker_msg(msg) } else { msgs::marker_array(msg) }.map_err(e)?;
            format!("{} markers{}", ms.len(), ms.first().map(|m| format!(", frame {}", m.frame_id)).unwrap_or_default())
        }
        Kind::Point => {
            let (h, p) = msgs::point_stamped(msg).map_err(e)?;
            format!("[{:.2} {:.2} {:.2}], frame {}", p[0], p[1], p[2], h.frame_id)
        }
        Kind::RobotDesc => format!("URDF, {} bytes", msgs::string(msg).map_err(e)?.len()),
    })
}
