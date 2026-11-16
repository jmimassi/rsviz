//! Bag playback: one thread reads the open bag and feeds the same handlers as the live subscriptions, paced by the
//! recorded receive times. The timeline (UI) drives it through `Playback`: play / pause, speed, loop and seek.
//!
//! A seek to any time rebuilds the scene as it was then, reading only what that needs: all static transforms, the last
//! seconds of /tf, the last message of each cloud / image / path, the recent markers, and the trails up to that time.
//! Dragging the cursor forward by a little just plays the messages in between.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::bag::Bag;
use crate::msgs;
use crate::ros::{Handler, Kind};
use crate::scene::{LineData, SceneData, Shared};

/// One topic of the bag, for the timeline.
pub struct TopicInfo {
    pub name: String,
    pub type_name: String,
    /// Display kind when the topic is shown by rsviz, None for the other topics of the bag.
    pub kind: Option<&'static str>,
    /// Receive time of every message, sorted (nanoseconds).
    pub times: Vec<u64>,
}

pub struct Ctl {
    pub playing: bool,
    pub speed: f64,
    pub looping: bool,
    /// Current bag time, nanoseconds.
    pub cur: u64,
    /// When `cur` was set (the timeline carries the cursor forward from there while playing).
    pub cur_at: Instant,
    /// Requested jump, taken by the player thread.
    pub seek: Option<u64>,
}

pub struct Playback {
    pub name: String,
    pub format: String,
    pub start: u64,
    pub end: u64,
    pub messages: usize,
    pub topics: Vec<TopicInfo>,
    pub ctl: Mutex<Ctl>,
    pub cv: Condvar,
}

impl Playback {
    pub fn seek(&self, t: u64) {
        self.ctl.lock().unwrap().seek = Some(t.clamp(self.start, self.end));
        self.cv.notify_all();
    }

    /// Jump by a signed number of seconds from the current (or already requested) time.
    pub fn step(&self, seconds: f64) {
        let mut c = self.ctl.lock().unwrap();
        let from = c.seek.unwrap_or(c.cur) as i128;
        let t = (from + (seconds * 1e9) as i128).clamp(self.start as i128, self.end as i128) as u64;
        c.seek = Some(t);
        drop(c);
        self.cv.notify_all();
    }

    pub fn toggle(&self) {
        let mut c = self.ctl.lock().unwrap();
        c.playing = !c.playing;
        if c.playing && c.cur >= self.end {
            c.seek = Some(self.start);
        }
        drop(c);
        self.cv.notify_all();
    }

    pub fn set_speed(&self, speed: f64) {
        self.ctl.lock().unwrap().speed = speed;
        self.cv.notify_all();
    }

    pub fn set_looping(&self, looping: bool) {
        self.ctl.lock().unwrap().looping = looping;
    }
}

#[derive(Clone)]
pub struct Config {
    /// Only these topics (all supported topics when None).
    pub topics: Option<Vec<String>>,
    /// Max positions kept per Odometry / PoseStamped trail.
    pub trail: usize,
    pub speed: f64,
    pub looping: bool,
    pub paused: bool,
    /// Where playback starts, seconds from the start of the bag.
    pub start: f64,
}

/// A file picked inside a rosbag2 directory (next to its metadata.yaml) stands for the whole recording.
fn resolve(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for p in paths {
        let dir = p.parent().filter(|d| p.is_file() && d.join("metadata.yaml").is_file()).map(|d| d.to_path_buf());
        let p = dir.unwrap_or(p);
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

fn label(paths: &[PathBuf]) -> String {
    let first = paths.first().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if paths.len() > 1 { format!("{first} (+{} files)", paths.len() - 1) } else { first }
}

/// Empties the scene. The version counter keeps growing, so that what the renderer cached is seen as outdated.
fn clear_scene(shared: &Shared) {
    let mut s = shared.scene.lock().unwrap();
    let (version, tf_version) = (s.version + 1, s.tf.version + 1);
    *s = SceneData::default();
    s.version = version;
    s.tf.version = tf_version;
}

/// Stops the thread that plays (or is opening) a bag and waits for it.
fn stop(shared: &Shared, old: Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)>) {
    if let Some((cancel, thread)) = old {
        cancel.store(true, Ordering::Release);
        if let Some(pb) = shared.player.lock().unwrap().clone() {
            pb.cv.notify_all();
        }
        let _ = thread.join();
        *shared.player.lock().unwrap() = None;
    }
}

/// Opens a bag (in the background) and plays it in place of the live topics, or of the bag that was open. What was
/// shown before stays as it is until the new bag is indexed, and for good if it cannot be opened.
pub fn open(shared: &Arc<Shared>, paths: Vec<PathBuf>, cfg: Config) {
    let paths = resolve(paths);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut slot = shared.player_thread.lock().unwrap();
    let old = slot.take();
    let (sh, flag) = (shared.clone(), cancel.clone());
    let thread = std::thread::Builder::new()
        .name("bag".into())
        .spawn(move || {
            let name = label(&paths);
            *sh.status.lock().unwrap() = crate::scene::Status { loading: Some(format!("Opening {name}")), notice: None };
            sh.waker.wake();
            let t0 = Instant::now();
            let res = Bag::open(&paths).and_then(|b| if b.entries.is_empty() { Err("the bag has no message".into()) } else { Ok(b) });
            sh.status.lock().unwrap().loading = None;
            match res {
                Ok(bag) => {
                    log::info!("{name}: {}, {} messages, {} topics, indexed in {:.0} ms", bag.format, bag.entries.len(), bag.topics.len(), t0.elapsed().as_secs_f64() * 1e3);
                    stop(&sh, old);
                    if !flag.load(Ordering::Acquire) {
                        run(sh, bag, name, cfg, flag);
                    }
                }
                Err(e) => {
                    log::error!("{name}: {e}");
                    sh.status.lock().unwrap().notice = Some(format!("{name}: {e}"));
                    // The bag that was playing goes on, unless a newer command took over meanwhile.
                    let mut slot = sh.player_thread.lock().unwrap();
                    if slot.as_ref().is_some_and(|(c, _)| Arc::ptr_eq(c, &flag)) {
                        *slot = old;
                    } else {
                        drop(slot);
                        stop(&sh, old);
                    }
                    // No bag at all (the one given on the command line failed): the live topics are shown.
                    if sh.player.lock().unwrap().is_none() {
                        sh.live.store(true, Ordering::Release);
                    }
                    sh.waker.wake();
                }
            }
        })
        .expect("spawn bag thread");
    *slot = Some((cancel, thread));
}

/// Closes the open bag and goes back to the live topics.
pub fn close(shared: &Arc<Shared>) {
    let old = shared.player_thread.lock().unwrap().take();
    let sh = shared.clone();
    std::thread::spawn(move || {
        stop(&sh, old);
        clear_scene(&sh);
        sh.live.store(true, Ordering::Release);
        sh.waker.wake();
    });
}

/// File dialog (zenity), then `open`. Without zenity, bags can still be dropped on the window or given as arguments.
pub fn pick(shared: &Arc<Shared>, cfg: Config) {
    let sh = shared.clone();
    std::thread::spawn(move || {
        let out = std::process::Command::new("zenity")
            .args(["--file-selection", "--multiple", "--separator=\n", "--title=Open a bag (ROS 1 .bag, ROS 2 .mcap / .db3)"])
            .args(["--file-filter=Bags | *.bag *.mcap *.db3 *.active", "--file-filter=All files | *"])
            .output();
        match out {
            Ok(o) => {
                let paths: Vec<PathBuf> = String::from_utf8_lossy(&o.stdout).lines().filter(|l| !l.is_empty()).map(PathBuf::from).collect();
                if !paths.is_empty() {
                    open(&sh, paths, cfg);
                }
            }
            Err(_) => {
                sh.status.lock().unwrap().notice = Some("No file dialog (zenity) on this machine: drop a bag on the window, or run rsviz <bag>".into());
                sh.waker.wake();
            }
        }
    });
}

struct Player {
    shared: Arc<Shared>,
    bag: Bag,
    pb: Arc<Playback>,
    cancel: Arc<AtomicBool>,
    trail: usize,
    kinds: Vec<Option<Kind>>,
    handlers: Vec<Option<Handler>>,
    /// Per topic, the indices of its entries (sorted by time).
    per_topic: Vec<Vec<u32>>,
    /// Topics not played message by message: not shown, or applied once (static transforms, robot model).
    skip: Vec<bool>,
    /// Scratch for `emit_range`: per topic, the last entry of the range.
    last: Vec<usize>,
    failed: Vec<bool>,
    /// Every static transform of the bag, applied whatever the time.
    statics: Vec<msgs::Transform>,
    /// Per topic, the entry whose message is the one shown now (usize::MAX = none). A seek that lands on the same
    /// message of a cloud, image or path does not read and upload it again (a static 200 MB map stays on the GPU).
    shown: Vec<usize>,
    /// Trails still being loaded after a seek (older positions are added in the background, newest first).
    pending: Vec<Trail>,
}

/// Part of a trail not loaded yet: the messages `per_topic[ti][lo..next]`.
#[derive(Clone, Copy)]
struct Trail {
    ti: usize,
    lo: usize,
    next: usize,
}

fn run(shared: Arc<Shared>, bag: Bag, name: String, cfg: Config, cancel: Arc<AtomicBool>) {
    let n = bag.topics.len();
    let kinds: Vec<Option<Kind>> = bag
        .topics
        .iter()
        .map(|t| Kind::of_topic(&t.name, &t.type_name).filter(|_| cfg.topics.as_ref().map_or(true, |f| f.contains(&t.name))))
        .collect();
    let mut per_topic: Vec<Vec<u32>> = bag.topics.iter().map(|t| Vec::with_capacity(t.count as usize)).collect();
    for (i, e) in bag.entries.iter().enumerate() {
        per_topic[e.topic as usize].push(i as u32);
    }
    let topics = bag
        .topics
        .iter()
        .enumerate()
        .map(|(ti, t)| TopicInfo {
            name: t.name.clone(),
            type_name: t.type_name.clone(),
            kind: kinds[ti].map(|k| k.label()),
            times: per_topic[ti].iter().map(|i| bag.entries[*i as usize].t).collect(),
        })
        .collect();
    let (start, end) = (bag.start(), bag.end());
    let pb = Arc::new(Playback {
        name,
        format: bag.format.clone(),
        start,
        end,
        messages: bag.entries.len(),
        topics,
        ctl: Mutex::new(Ctl { playing: !cfg.paused, speed: cfg.speed, looping: cfg.looping, cur: start, cur_at: Instant::now(), seek: Some(start.saturating_add((cfg.start.max(0.0) * 1e9) as u64).min(end)) }),
        cv: Condvar::new(),
    });

    shared.live.store(false, Ordering::Release);
    clear_scene(&shared);
    let handlers: Vec<Option<Handler>> =
        kinds.iter().enumerate().map(|(ti, k)| k.map(|k| Handler::new(shared.clone(), bag.topics[ti].name.clone(), k, cfg.trail, "bag".into(), true))).collect();
    if !bag.warnings.is_empty() {
        shared.status.lock().unwrap().notice = Some(bag.warnings.join(" · "));
    }
    *shared.player.lock().unwrap() = Some(pb.clone());
    let skip = handlers.iter().map(|h| h.is_none()).collect();
    let mut p = Player { shared, bag, pb, cancel, trail: cfg.trail, kinds, handlers, per_topic, skip, last: vec![usize::MAX; n], failed: vec![false; n], statics: Vec::new(), shown: vec![usize::MAX; n], pending: Vec::new() };
    p.load_statics();
    p.play();
}

impl Player {
    /// A new command is waiting: whatever is being rebuilt or emitted is abandoned for it.
    fn interrupted(&self) -> bool {
        self.cancel.load(Ordering::Acquire) || self.pb.ctl.lock().unwrap().seek.is_some()
    }

    fn fail(&mut self, ti: usize, err: String) {
        if !std::mem::replace(&mut self.failed[ti], true) {
            let name = &self.bag.topics[ti].name;
            log::warn!("{name}: {err}");
            self.shared.scene.lock().unwrap().stats.entry(name.clone()).or_default().error = Some(err);
        }
    }

    fn read(&mut self, k: usize) -> Option<Vec<u8>> {
        let e = self.bag.entries[k];
        match self.bag.read(&e) {
            Ok(m) => Some(m),
            Err(err) => {
                self.fail(e.topic as usize, err);
                None
            }
        }
    }

    fn emit(&mut self, k: usize) {
        let ti = self.bag.entries[k].topic as usize;
        if let Some(msg) = self.read(k) {
            if let Some(h) = &self.handlers[ti] {
                h.handle(msg);
                self.shown[ti] = k;
            }
        }
    }

    /// Plays entries [i, j) and returns j, or the entry it stopped at when interrupted. Of a topic whose last message
    /// is its whole state (image, cloud, path) only the last message of the range is read.
    fn emit_range(&mut self, i: usize, j: usize) -> usize {
        for k in i..j {
            let ti = self.bag.entries[k].topic as usize;
            if !self.skip[ti] {
                self.last[ti] = k;
            }
        }
        for k in i..j {
            if (k - i) % 256 == 255 && self.interrupted() {
                return k;
            }
            let ti = self.bag.entries[k].topic as usize;
            if self.skip[ti] || (self.last[ti] != k && self.kinds[ti].is_some_and(|kind| kind.latest_only())) {
                continue;
            }
            self.emit(k);
        }
        j
    }

    /// Static transforms and robot models hold for the whole bag: read once, never replayed.
    fn load_statics(&mut self) {
        for ti in 0..self.kinds.len() {
            match self.kinds[ti] {
                Some(Kind::Tf) if self.bag.topics[ti].name.ends_with("tf_static") => {
                    self.skip[ti] = true;
                    let idx: Vec<u32> = self.per_topic[ti].iter().take(20_000).copied().collect();
                    for k in idx {
                        if let Some(msg) = self.read(k as usize) {
                            match msgs::tf_message(&msg) {
                                Ok(tfs) => self.statics.extend(tfs),
                                Err(e) => self.fail(ti, e.0.to_string()),
                            }
                        }
                    }
                }
                Some(Kind::RobotDesc) => {
                    self.skip[ti] = true;
                    if let Some(k) = self.per_topic[ti].first().copied() {
                        self.emit(k as usize);
                    }
                }
                _ => {}
            }
        }
    }

    /// Entries of topic `ti` before entry index `j`, newest first.
    fn before(&self, ti: usize, j: usize) -> impl Iterator<Item = usize> + '_ {
        let idx = &self.per_topic[ti];
        let k = idx.partition_point(|x| (*x as usize) < j);
        idx[..k].iter().rev().map(|x| *x as usize)
    }

    /// The last messages of topic `ti` before entry `j`, oldest first: those of the last `span` nanoseconds before `t`
    /// and in any case the last `min` (a marker or transform published once, long ago, is kept), at most `max`, and no
    /// more than can be read within `budget`.
    fn read_back(&mut self, ti: usize, j: usize, t: u64, span: u64, min: usize, max: usize, budget: Duration) -> Vec<Vec<u8>> {
        let t0 = Instant::now();
        let ks: Vec<usize> = self.before(ti, j).take(max).collect();
        let mut out = Vec::new();
        for (n, k) in ks.into_iter().enumerate() {
            let old = self.bag.entries[k].t.saturating_add(span) < t && n >= min;
            if old || (n % 16 == 15 && t0.elapsed() > budget) {
                break;
            }
            if let Some(m) = self.read(k) {
                out.push(m);
            }
        }
        out.reverse();
        out
    }

    /// Rebuilds the scene as of entries [0, j) (bag time `t`). Returns false when interrupted by a newer command.
    /// What the views need at once comes first (transforms, last image / cloud / path, recent markers); the trails
    /// get their newest part, and the rest is left to `refine`.
    fn rebuild(&mut self, j: usize, t: u64) -> bool {
        self.pending.clear();
        {
            let mut s = self.shared.scene.lock().unwrap();
            s.tf.clear();
            for tr in &self.statics {
                s.tf.set(&tr.parent, &tr.child, tr.t, tr.q);
            }
            for (ti, kind) in self.kinds.iter().enumerate() {
                let name = &self.bag.topics[ti].name;
                match kind {
                    Some(Kind::Marker | Kind::MarkerArray | Kind::Point) => {
                        s.markers.remove(name);
                    }
                    Some(Kind::Odom | Kind::Pose) => {
                        s.lines.remove(name);
                    }
                    _ => {}
                }
            }
            s.version += 1;
        }
        for ti in 0..self.kinds.len() {
            let Some(kind) = self.kinds[ti].filter(|_| !self.skip[ti]) else { continue };
            if self.interrupted() {
                return false;
            }
            match kind {
                Kind::Tf | Kind::Marker | Kind::MarkerArray => {
                    let msgs = if kind == Kind::Tf {
                        self.read_back(ti, j, t, 3_000_000_000, 200, 6000, Duration::from_millis(150))
                    } else {
                        self.read_back(ti, j, t, 2_000_000_000, 60, 300, Duration::from_millis(100))
                    };
                    if let Some(h) = &self.handlers[ti] {
                        for m in msgs {
                            h.handle(m);
                        }
                    }
                }
                Kind::Cloud | Kind::Image | Kind::Compressed | Kind::Path | Kind::Point => {
                    let last = self.before(ti, j).next();
                    match last {
                        Some(k) if self.shown[ti] == k => {}
                        Some(k) => self.emit(k),
                        None => {
                            self.shown[ti] = usize::MAX;
                            // Before the first message of the topic. An image stays (its tile would close otherwise).
                            let mut s = self.shared.scene.lock().unwrap();
                            let name = &self.bag.topics[ti].name;
                            s.clouds.remove(name);
                            s.lines.remove(name);
                            s.version += 1;
                        }
                    }
                }
                Kind::Odom | Kind::Pose => {
                    let end = self.per_topic[ti].partition_point(|x| (*x as usize) < j);
                    if end > 0 {
                        self.pending.push(Trail { ti, lo: end.saturating_sub(self.trail), next: end });
                    }
                }
                Kind::RobotDesc => {}
            }
        }
        self.shared.waker.wake();
        self.refine(Instant::now() + Duration::from_millis(60));
        true
    }

    /// Loads more of the pending trails until `deadline`: chunks of older positions, the topics in turn, so that
    /// every trail grows backwards from the current time. Runs between two commands, never delays one for long.
    fn refine(&mut self, deadline: Instant) {
        while let Some(tr) = self.pending.first().copied() {
            if Instant::now() >= deadline || self.cancel.load(Ordering::Acquire) {
                return;
            }
            self.pending.remove(0);
            let a = tr.next.saturating_sub(256).max(tr.lo);
            let ks: Vec<usize> = self.per_topic[tr.ti][a..tr.next].iter().map(|x| *x as usize).collect();
            let odom = self.kinds[tr.ti] == Some(Kind::Odom);
            let mut frame_id = None;
            let mut points = Vec::with_capacity(ks.len());
            for k in ks {
                let Some(msg) = self.read(k) else { continue };
                match if odom { msgs::odometry(&msg) } else { msgs::pose_stamped(&msg) } {
                    Ok((h, p, _)) => {
                        frame_id = Some(h.frame_id);
                        points.push(p);
                    }
                    Err(e) => self.fail(tr.ti, e.0.to_string()),
                }
            }
            let full = {
                let mut s = self.shared.scene.lock().unwrap();
                s.version += 1;
                let version = s.version;
                let l = s.lines.entry(self.bag.topics[tr.ti].name.clone()).or_insert_with(|| LineData { frame_id: String::new(), points: Default::default(), version });
                if let Some(f) = frame_id.filter(|_| l.frame_id.is_empty()) {
                    l.frame_id = f;
                }
                // Older than everything already there (the player only appends newer positions meanwhile).
                let room = self.trail.saturating_sub(l.points.len());
                for p in points.into_iter().rev().take(room) {
                    l.points.push_front(p);
                }
                l.version = version;
                l.points.len() >= self.trail
            };
            if a > tr.lo && !full {
                self.pending.push(Trail { next: a, ..tr });
            }
            self.shared.waker.wake();
        }
    }

    /// Time of the next entry that is actually played, from entry `i` (looks a bounded number of entries ahead).
    fn next_time(&self, i: usize) -> Option<u64> {
        self.bag.entries[i..].iter().take(4096).find(|e| !self.skip[e.topic as usize]).map(|e| e.t)
    }

    fn play(&mut self) {
        let n = self.bag.entries.len();
        let (start, end) = (self.pb.start, self.pb.end);
        let pb = self.pb.clone();
        // Next entry to play; `dirty` = the scene does not match it (first pass, or a rebuild that was interrupted).
        let (mut i, mut dirty) = (0usize, true);
        // While playing: (wall clock, bag time, speed) the bag time is extrapolated from.
        let mut anchor: Option<(Instant, u64, f64)> = None;
        let mut last_tick = Instant::now();
        loop {
            if self.cancel.load(Ordering::Acquire) {
                return;
            }
            let seek = {
                let mut c = pb.ctl.lock().unwrap();
                c.seek.take().map(|t| (t, c.cur))
            };
            if let Some((t, from)) = seek {
                let j = self.bag.entries.partition_point(|e| e.t <= t);
                let t0 = Instant::now();
                let forward = !dirty && j >= i && j - i <= 20_000 && t >= from;
                if forward {
                    i = self.emit_range(i, j);
                } else {
                    dirty = !self.rebuild(j, t);
                    i = j;
                }
                log::debug!("seek to {:.3} s: {} in {:.1} ms", t.saturating_sub(start) as f64 * 1e-9, if forward { "played forward" } else { "scene rebuilt" }, t0.elapsed().as_secs_f64() * 1e3);
                {
                    let mut c = pb.ctl.lock().unwrap();
                    c.cur = t;
                    c.cur_at = Instant::now();
                }
                anchor = None;
                self.shared.waker.wake();
                continue;
            }
            // Trails cut short by the last seek are completed here, a few milliseconds at a time.
            let refining = !self.pending.is_empty();
            if refining {
                self.refine(Instant::now() + Duration::from_millis(5));
            }
            let c = pb.ctl.lock().unwrap();
            if c.seek.is_some() {
                continue;
            }
            if !c.playing {
                anchor = None;
                if !refining {
                    drop(pb.cv.wait_timeout(c, Duration::from_millis(200)).unwrap());
                }
                continue;
            }
            let now = Instant::now();
            let (wall, t0, speed) = match anchor {
                Some(a) if a.2 == c.speed => a,
                _ => {
                    let a = (now, c.cur, c.speed.max(1e-3));
                    anchor = Some(a);
                    a
                }
            };
            drop(c);
            let target = t0.saturating_add((now.duration_since(wall).as_secs_f64() * speed * 1e9) as u64).min(end);
            let j = i + self.bag.entries[i..].partition_point(|e| e.t <= target);
            let reached = self.emit_range(i, j);
            i = reached;
            if reached < j {
                continue; // interrupted: the pending command is handled first
            }
            let mut c = pb.ctl.lock().unwrap();
            if c.seek.is_some() {
                continue;
            }
            c.cur = target;
            c.cur_at = now;
            if target >= end && i >= n {
                if c.looping && end > start {
                    c.seek = Some(start);
                } else {
                    c.playing = false;
                }
                drop(c);
                self.shared.waker.wake();
                continue;
            }
            if last_tick.elapsed() >= Duration::from_millis(33) {
                last_tick = Instant::now();
                self.shared.waker.wake(); // the cursor moves even when no message is due
            }
            if !self.pending.is_empty() {
                continue; // more trail to load: no idle wait
            }
            let until_next = self.next_time(i).map_or(0.033, |t| t.saturating_sub(target) as f64 * 1e-9 / speed);
            let wait = Duration::from_secs_f64(until_next.clamp(0.001, 0.033));
            drop(pb.cv.wait_timeout(c, wait).unwrap());
        }
    }
}

/// `rsviz --bag-info <bag>`: prints what the bag holds and decodes the first, middle and last message of every topic
/// rsviz shows (a check of the reader and of the decoders, without any window). Returns the process exit code.
pub fn info(paths: Vec<PathBuf>) -> i32 {
    let paths = resolve(paths);
    let t0 = Instant::now();
    let mut bag = match Bag::open(&paths) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let dur = bag.end().saturating_sub(bag.start()) as f64 * 1e-9;
    println!("{}: {}, {} messages, {} topics, {:.3} s, indexed in {:.0} ms", label(&paths), bag.format, bag.entries.len(), bag.topics.len(), dur, t0.elapsed().as_secs_f64() * 1e3);
    for w in &bag.warnings {
        println!("warning: {w}");
    }
    let mut first_last: Vec<Vec<usize>> = vec![Vec::new(); bag.topics.len()];
    for (i, e) in bag.entries.iter().enumerate() {
        first_last[e.topic as usize].push(i);
    }
    let mut errors = 0;
    let mut order: Vec<usize> = (0..bag.topics.len()).collect();
    order.sort_by(|a, b| bag.topics[*a].name.cmp(&bag.topics[*b].name));
    for ti in order {
        let (name, ty, count) = (bag.topics[ti].name.clone(), bag.topics[ti].type_name.clone(), bag.topics[ti].count);
        let Some(kind) = Kind::of_topic(&name, &ty) else {
            println!("  {name:<58} {count:>8}  {ty}");
            continue;
        };
        let idx = &first_last[ti];
        let mut picks = vec![idx[0], idx[idx.len() / 2], idx[idx.len() - 1]];
        picks.dedup();
        let mut note = String::new();
        for k in picks {
            let e = bag.entries[k];
            match bag.read(&e).and_then(|m| crate::ros::describe(kind, &m)) {
                Ok(d) => note = d,
                Err(err) => {
                    note = format!("ERROR {err}");
                    errors += 1;
                    break;
                }
            }
        }
        println!("* {name:<58} {count:>8}  {:<16} {note}", kind.label());
    }
    println!("(* = shown by rsviz)");
    if errors > 0 { 2 } else { 0 }
}
