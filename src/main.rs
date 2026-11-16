//! rsviz: a small, efficient ROS 2 viewer. 3D view (point clouds, paths, odometry / pose trails, TF frames, grid) and
//! camera panels (Image, CompressedImage). Native ROS 2 node, no bridge. It draws at the refresh rate of the display
//! (vsync); with --on-demand it draws only when a message arrives or the user interacts (near zero load when idle).
//!
//! It also plays bags, without ROS: ROS 1 (.bag) and ROS 2 (.mcap, .db3, or a rosbag2 directory), with a timeline
//! (play / pause, speed, loop, seek, one lane of message density per topic). While a bag is open the live topics are
//! muted; closing it brings them back.
//!
//!   rsviz [bag ...] [--fixed-frame map] [--follow base_link|none] [--distance 30] [--pitch 0.6] [--topics /a,/b]
//!         [--point-size 2] [--line-width 2] [--trail 20000] [--width 1600 --height 1000] [--present fifo|mailbox|immediate] [--max-fps N]
//!         [--rate 1.0] [--start 0] [--loop] [--paused] [--bag-info] [--on-demand] [--gpu intel|nvidia] [--fullscreen] [--domain N]
//!   bag ...: the bag to play (several files = the splits of one recording). Also: Open bag, or drop it on the window.
//!   --rate, --start, --loop, --paused: playback speed, start time (seconds into the bag), loop at the end, start paused.
//!   --domain: ROS domain to listen on (default: ROS_DOMAIN_ID); it can also be changed from the top bar.
//!   --bag-info: no window; prints the topics of the bag and checks that the shown ones decode.
//!   --on-demand: draw only when something changes (default: every refresh of the display).
//!   --fullscreen: start in full screen on the primary monitor (F11 toggles it on the current one). A full-screen
//!          window is not recomposed by the desktop, which lifts the frame rate where the compositor is the limit.
//!   --gpu: the GPU to draw with, by part of its name (default: the most powerful one). On a laptop whose screens
//!          are driven by the integrated GPU, drawing with that one avoids the copy of every frame between GPUs.
//!   --follow: the camera orbits this TF frame (default: base_link as soon as it exists; none = fixed point).
//!   RSVIZ_MESH_PATH=dir1:dir2: where to look (by file name) for robot meshes whose path does not exist here.

mod bag;
mod cdr;
mod gfx;
mod msgs;
mod player;
mod ros;
mod scene;
mod shapes;
mod ui;
mod urdf;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{DAffine3, Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use gfx::{Camera, CloudSnapshot, Gpu, ImageTex, LineVertex, MeshSnapshot, View3D};
use scene::{Prim, SceneData, Shared};

struct Args {
    fixed_frame: String,
    follow: Option<String>,
    distance: f32,
    pitch: f32,
    topics: Option<Vec<String>>,
    point_px: f32,
    line_px: f32,
    trail: usize,
    width: u32,
    height: u32,
    /// None = chosen from the drawing mode (see `resumed`).
    present: Option<wgpu::PresentMode>,
    max_fps: Option<f32>,
    bags: Vec<std::path::PathBuf>,
    rate: f64,
    looping: bool,
    paused: bool,
    bag_info: bool,
    on_demand: bool,
    gpu: Option<String>,
    fullscreen: bool,
    start: f64,
}

impl Args {
    fn bag_config(&self) -> player::Config {
        player::Config { topics: self.topics.clone(), trail: self.trail, speed: self.rate, looping: self.looping, paused: self.paused, start: self.start }
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        fixed_frame: "map".into(),
        follow: Some(String::new()),
        distance: 30.0,
        pitch: 0.6,
        topics: None,
        point_px: 2.0,
        line_px: 2.0,
        trail: 20000,
        width: 1600,
        height: 1000,
        present: None,
        max_fps: None,
        bags: Vec::new(),
        rate: 1.0,
        looping: false,
        paused: false,
        bag_info: false,
        on_demand: false,
        gpu: None,
        fullscreen: false,
        start: 0.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        match k.as_str() {
            "--fixed-frame" => a.fixed_frame = v(),
            "--follow" => a.follow = Some(v()).filter(|f| f != "none"),
            "--distance" => a.distance = v().parse().expect("--distance"),
            "--pitch" => a.pitch = v().parse().expect("--pitch"),
            "--topics" => a.topics = Some(v().split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect()),
            "--point-size" => a.point_px = v().parse().expect("--point-size"),
            "--line-width" => a.line_px = v().parse().expect("--line-width"),
            "--trail" => a.trail = v().parse().expect("--trail"),
            "--width" => a.width = v().parse().expect("--width"),
            "--height" => a.height = v().parse().expect("--height"),
            "--max-fps" => a.max_fps = Some(v().parse().expect("--max-fps")),
            "--rate" => a.rate = v().parse::<f64>().expect("--rate").clamp(0.01, 100.0),
            "--loop" => a.looping = true,
            "--paused" => a.paused = true,
            "--bag-info" => a.bag_info = true,
            "--on-demand" => a.on_demand = true,
            "--gpu" => a.gpu = Some(v()),
            "--fullscreen" => a.fullscreen = true,
            "--domain" => std::env::set_var("ROS_DOMAIN_ID", v()),
            "--start" => a.start = v().parse().expect("--start"),
            "--present" => {
                a.present = Some(match v().as_str() {
                    "immediate" => wgpu::PresentMode::Immediate,
                    "mailbox" => wgpu::PresentMode::Mailbox,
                    _ => wgpu::PresentMode::Fifo,
                })
            }
            "-h" | "--help" => {
                println!("{}", include_str!("main.rs").lines().take_while(|l| l.starts_with("//!")).map(|l| l.trim_start_matches("//!")).collect::<Vec<_>>().join("\n"));
                std::process::exit(0);
            }
            _ if !k.starts_with("--") => a.bags.push(k.clone().into()),
            _ => panic!("unknown argument {k} (see --help)"),
        }
    }
    a
}

/// Categorical colours for line topics (fixed order, never cycled through random hues).
const PALETTE: [[f32; 4]; 6] = [
    [0.16, 0.47, 0.84, 1.0],
    [0.92, 0.41, 0.20, 1.0],
    [0.11, 0.69, 0.48, 1.0],
    [0.93, 0.63, 0.00, 1.0],
    [0.91, 0.48, 0.64, 1.0],
    [0.38, 0.31, 0.84, 1.0],
];

struct Running {
    window: Arc<Window>,
    gpu: Gpu,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_rend: egui_wgpu::Renderer,
    view: View3D,
    images: HashMap<String, Option<ImageTex>>,
    camera: Camera,
    visible: HashMap<String, bool>,
    /// Per-topic settings of the topics panel, and a counter of their changes (the lines are rebuilt on a change).
    styles: HashMap<String, ui::TopicStyle>,
    tf_style: ui::TfStyle,
    style_rev: u64,
    lines_key: (u64, String, bool, bool, [i64; 3], i32, u64),
    settings: ui::ViewSettings,
    tree: egui_tiles::Tree<ui::Pane>,
    tree_key: Vec<String>,
    timeline: ui::TimelineState,
    /// The bag whose frames were already looked at to pick a fixed frame (address of its playback, 0 = none).
    framed_bag: usize,
    frame_times: std::collections::VecDeque<Instant>,
    /// Smoothed cost of one frame in ms (work only: the waits on the display are left out).
    frame_ms: f32,
    /// Smoothed time spent per frame waiting for the display (acquire + present), ms.
    wait_ms: f32,
    last_frame: Instant,
    /// Refresh rate of the monitor the window is on (Hz), looked up once per second.
    refresh: f32,
    refresh_at: Instant,
    /// Vsync does not hold the loop back (present mode without vsync, or seen so on this display): continuous
    /// drawing is paced at `refresh` by the clock.
    limit: bool,
}

struct App {
    args: Args,
    shared: Arc<Shared>,
    run: Option<Running>,
}

impl App {
    fn redraw(&mut self) {
        let Some(r) = self.run.as_mut() else { return };
        self.shared.waker.clear();
        let now = Instant::now();
        r.frame_times.push_back(now);
        while r.frame_times.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(2)) {
            r.frame_times.pop_front();
        }
        let fps = r.frame_times.len() as f32 / 2.0;
        let (frame_ms, wait_ms, refresh) = (r.frame_ms, r.wait_ms, r.refresh);
        let frame_start = now;
        // Time since the previous frame was done (and paced): the event loop getting back to us.
        let gap = frame_start.saturating_duration_since(r.last_frame);

        // Light copy of what the UI shows (no message data).
        let (rows, image_meta, frames, texts, tf_root) = {
            let mut s = self.shared.scene.lock().unwrap();
            // Markers past their lifetime.
            let now = Instant::now();
            let mut expired = false;
            for m in s.markers.values_mut() {
                let before = m.len();
                m.retain(|_, d| d.expires.map_or(true, |e| e > now));
                expired |= m.len() != before;
            }
            if expired {
                s.version += 1;
            }
            let mut texts = Vec::new();
            for (topic, ms) in &s.markers {
                if !r.visible.get(topic).copied().unwrap_or(true) {
                    continue;
                }
                for d in ms.values() {
                    if let Prim::Text { pos, text, color } = &d.prim {
                        if let Some(a) = s.tf.lookup(&self.args.fixed_frame, &d.frame_id) {
                            let w = (a * d.pose).transform_point3(glam::Vec3::from(*pos).as_dvec3()).as_vec3();
                            texts.push((w, text.clone(), *color));
                        }
                    }
                }
            }
            let mut rows: Vec<ui::TopicRow> = s
                .stats
                .iter()
                .map(|(k, v)| ui::TopicRow { name: k.clone(), kind: v.kind, hz: v.hz(), qos: v.qos.clone(), error: v.error.clone() })
                .collect();
            rows.sort_by(|a, b| a.name.cmp(&b.name));
            let im: BTreeMap<String, (u32, u32, f32)> = s.images.iter().map(|(k, v)| (k.clone(), (v.width, v.height, v.decode_ms))).collect();
            (rows, im, s.tf.frames(), texts, s.tf.root())
        };
        // New topics are shown, except a bag's raw camera topic when its relayed twin exists (".../image_raw/..." vs ".../image/...").
        let names: std::collections::HashSet<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        for row in &rows {
            let twin = row.name.contains("/image_raw/") && names.contains(row.name.replace("/image_raw/", "/image/").as_str());
            r.visible.entry(row.name.clone()).or_insert(!twin);
        }

        // Upload changed images before the UI references their textures.
        {
            let s = self.shared.scene.lock().unwrap();
            for (topic, img) in &s.images {
                if !r.visible.get(topic).copied().unwrap_or(true) {
                    continue;
                }
                let slot = r.images.entry(topic.clone()).or_insert(None);
                if slot.as_ref().map_or(true, |t| t.version != img.version) {
                    let rgba = img.rgba.clone();
                    ImageTex::update(slot, &r.gpu.device, &r.gpu.queue, &mut r.egui_rend, img.width, img.height, &rgba, img.version);
                }
            }
        }

        let mut view_px = (r.view.size.0, r.view.size.1);
        let vp_ui = r.camera.view_proj(r.view.size.0 as f32 / r.view.size.1.max(1) as f32);
        // Tiles: rebuilt only when the set of shown cameras changes (a moved tile stays where the user put it otherwise).
        let shown: Vec<String> = image_meta.keys().filter(|k| r.visible.get(*k).copied().unwrap_or(true)).cloned().collect();
        if shown != r.tree_key {
            r.tree = ui::build_tree(&shown);
            r.tree_key = shown;
        }
        let hz: HashMap<String, f32> = rows.iter().map(|row| (row.name.clone(), row.hz)).collect();
        let raw = r.egui_state.take_egui_input(&r.window);
        let bag_cfg = self.args.bag_config();
        let args = &mut self.args;
        let playback = self.shared.player.lock().unwrap().clone();
        let (loading, notice) = {
            let st = self.shared.status.lock().unwrap();
            (st.loading.clone(), st.notice.clone())
        };
        let source = match &playback {
            Some(pb) => format!("bag {}", pb.name),
            None => format!("ROS domain {}", self.shared.domain.load(std::sync::atomic::Ordering::Acquire)),
        };
        // A bag whose transforms do not have the current fixed frame would show an empty 3D view: once per bag, as
        // soon as its frames are known, the view moves to the root of its TF tree and is centred on the content.
        let bag_id = playback.as_ref().map_or(0, |pb| Arc::as_ptr(pb) as usize);
        let mut auto_focus = false;
        if bag_id != r.framed_bag && (bag_id == 0 || !frames.is_empty()) {
            r.framed_bag = bag_id;
            if bag_id != 0 {
                if let Some(root) = tf_root.filter(|_| !frames.contains(&args.fixed_frame)) {
                    log::info!("fixed frame {} is not in this bag: using {root}", args.fixed_frame);
                    args.fixed_frame = root;
                    r.window.request_redraw();
                }
                auto_focus = true;
            }
        }
        let mut top = ui::TopAction::default();
        let mut close_bag = false;
        let out = {
            let (visible, camera, settings, tree, images, timeline) = (&mut r.visible, &mut r.camera, &mut r.settings, &mut r.tree, &r.images, &mut r.timeline);
            let (styles, style_rev, tf_style) = (&mut r.styles, &mut r.style_rev, &mut r.tf_style);
            let view_id = r.view.egui_id;
            r.egui_ctx.run(raw, |ctx| {
                top = ui::top_bar(ctx, ui::TopBar { fixed_frame: &mut args.fixed_frame, follow: &mut args.follow, frames: &frames, fps, frame_ms, wait_ms, refresh, on_demand: args.on_demand, n_topics: rows.len(), source: source.clone(), domain: playback.is_none().then(|| self.shared.domain.load(std::sync::atomic::Ordering::Acquire)), loading: loading.clone(), notice: notice.clone() });
                if let Some(pb) = &playback {
                    close_bag = ui::timeline(ctx, pb, timeline);
                }
                if ui::topics_panel(ctx, &rows, visible, styles, (settings.line_px, settings.point_px), &frames, tf_style) {
                    *style_rev += 1;
                }
                egui::CentralPanel::default().frame(egui::Frame::none().fill(ui::pal::BG)).show(ctx, |ui| {
                    let mut b = ui::Behavior { view_tex: view_id, view_px: &mut view_px, camera, vp: vp_ui, texts: &texts, images, image_meta: &image_meta, hz: &hz, settings };
                    tree.ui(&mut b, ui);
                });
            })
        };
        r.egui_state.handle_platform_output(&r.window, out.platform_output);
        let t_ui = frame_start.elapsed();
        let focus_flag = top.focus || auto_focus;
        if top.open {
            player::pick(&self.shared, bag_cfg);
        }
        if let Some(d) = top.domain {
            // r2r keeps one rcl context for the whole process, created with the domain of the start: listening on
            // another domain is a restart of rsviz with the same arguments and --domain d.
            log::info!("restarting on ROS domain {d}");
            let mut argv: Vec<String> = Vec::new();
            let mut it = std::env::args().skip(1);
            while let Some(a) = it.next() {
                if a == "--domain" {
                    it.next();
                } else {
                    argv.push(a);
                }
            }
            argv.extend(["--domain".into(), d.to_string()]);
            use std::os::unix::process::CommandExt;
            let exe = std::env::current_exe().unwrap_or_else(|_| "rsviz".into());
            let e = std::process::Command::new(exe).args(argv).env("ROS_DOMAIN_ID", d.to_string()).exec();
            log::error!("cannot restart rsviz: {e}");
        }
        if top.dismiss {
            self.shared.status.lock().unwrap().notice = None;
        }
        if close_bag {
            r.timeline = ui::TimelineState::default();
            player::close(&self.shared);
        }

        // Camera anchor: the followed frame (auto = base_link once it exists), else the scene centre on Focus.
        {
            let s = self.shared.scene.lock().unwrap();
            let follow = match &self.args.follow {
                Some(f) if f.is_empty() => s.tf.lookup(&self.args.fixed_frame, "base_link").map(|_| "base_link".to_owned()),
                Some(f) => Some(f.clone()),
                None => None,
            };
            if let Some(a) = follow.as_deref().and_then(|f| s.tf.lookup(&self.args.fixed_frame, f)) {
                r.camera.anchor = a.translation.as_vec3();
                if focus_flag {
                    r.camera.offset = Vec3::ZERO;
                }
            } else if focus_flag {
                let mut pts: Vec<Vec3> = Vec::new();
                for l in s.lines.values() {
                    if let (Some(a), Some(p)) = (s.tf.lookup(&self.args.fixed_frame, &l.frame_id), l.points.back()) {
                        pts.push(a.transform_point3(glam::DVec3::from(*p)).as_vec3());
                    }
                }
                for ms in s.markers.values() {
                    for d in ms.values() {
                        if let Some(a) = s.tf.lookup(&self.args.fixed_frame, &d.frame_id) {
                            pts.push((a * d.pose).translation.as_vec3());
                        }
                    }
                }
                if !pts.is_empty() {
                    r.camera.anchor = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
                    r.camera.offset = Vec3::ZERO;
                }
            }
        }

        // 3D: target size, lines (rebuilt only when the scene, frame or toggles changed), clouds.
        r.view.ensure_size(&r.gpu.device, &mut r.egui_rend, view_px.0, view_px.1);
        let aspect = r.view.size.0 as f32 / r.view.size.1 as f32;
        let vp = r.camera.view_proj(aspect);
        let (snaps, mesh_snaps) = {
            let s = self.shared.scene.lock().unwrap();
            // Grid centred under the camera target, with a step that follows the zoom (1, 10, 100 m ...).
            let step = 10f32.powi((r.camera.dist / 4.0).max(0.1).log10().floor() as i32);
            let target = r.camera.anchor + r.camera.offset;
            let cell = [(target.x / step).round() as i64, (target.y / step).round() as i64, 0];
            let key = (s.version, self.args.fixed_frame.clone(), r.settings.grid, r.settings.axes, cell, step.log10().round() as i32, r.style_rev);
            if key != r.lines_key {
                let grid = r.settings.grid.then(|| (Vec3::new(cell[0] as f32 * step, cell[1] as f32 * step, 0.0), step));
                let verts = build_lines(&s, &self.args.fixed_frame, &r.visible, &r.styles, &r.tf_style, grid, r.settings.axes);
                r.view.set_lines(&r.gpu.device, &r.gpu.queue, &verts);
                r.lines_key = key;
            }
            let clouds = s
                .clouds
                .iter()
                .filter(|(k, _)| r.visible.get(*k).copied().unwrap_or(true))
                .filter_map(|(k, c)| {
                    let model = s.tf.lookup(&self.args.fixed_frame, &c.frame_id)?;
                    Some((
                        k.clone(),
                        CloudSnapshot {
                            bytes: c.bytes.clone(),
                            off: c.off,
                            len: c.len,
                            n: c.n,
                            point_step: c.point_step,
                            xyz: c.xyz,
                            color: c.color,
                            range: c.range,
                            chunks: c.chunks.clone(),
                            version: c.version,
                            model: to_mat4(&model),
                        },
                    ))
                })
                .collect::<HashMap<_, _>>();
            let mut meshes = HashMap::new();
            for (topic, ms) in &s.markers {
                if !r.visible.get(topic).copied().unwrap_or(true) {
                    continue;
                }
                for ((ns, id), d) in ms {
                    let Prim::Mesh { verts, texture, shade } = &d.prim else { continue };
                    let Some(a) = s.tf.lookup(&self.args.fixed_frame, &d.frame_id) else { continue };
                    meshes.insert(
                        format!("{topic}|{ns}|{id}"),
                        MeshSnapshot { verts: verts.clone(), texture: texture.clone(), shade: *shade, model: to_mat4(&(a * d.pose)), version: d.version },
                    );
                }
            }
            for (topic, robot) in &s.robots {
                if !r.visible.get(topic).copied().unwrap_or(true) {
                    continue;
                }
                for (i, v) in robot.visuals.iter().enumerate() {
                    let Some(a) = s.tf.lookup(&self.args.fixed_frame, &v.link) else { continue };
                    meshes.insert(format!("{topic}#{i}"), MeshSnapshot { verts: v.verts.clone(), texture: None, shade: true, model: to_mat4(&(a * v.origin)), version: robot.version });
                }
            }
            (clouds, meshes)
        };
        let topic_px: HashMap<String, f32> = r.styles.iter().filter_map(|(k, v)| v.point_px.map(|p| (k.clone(), p))).collect();
        r.view.sync_clouds(&r.gpu.device, &r.gpu.queue, &snaps, vp, r.camera.eye(), r.settings.point_px, &topic_px);
        r.view.sync_meshes(&r.gpu.device, &r.gpu.queue, &mesh_snaps, vp);

        let acquire_start = Instant::now();
        let frame = match r.gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                let (w, h) = (r.gpu.config.width, r.gpu.config.height);
                r.gpu.resize(w, h);
                r.window.request_redraw();
                return;
            }
            Err(e) => {
                log::warn!("surface: {e:?}");
                if !self.args.on_demand {
                    // Continuous drawing goes on once the surface answers again (hidden window, busy compositor).
                    std::thread::sleep(Duration::from_millis(50));
                    r.window.request_redraw();
                }
                return;
            }
        };
        // Under vsync the wait for the display happens in the acquire above and in the present below.
        let mut waited = acquire_start.elapsed();
        let (t_scene, t_acquire) = (acquire_start.duration_since(frame_start).saturating_sub(t_ui), waited);
        let target = frame.texture.create_view(&Default::default());
        let jobs = r.egui_ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [r.gpu.config.width, r.gpu.config.height], pixels_per_point: out.pixels_per_point };
        for (id, delta) in &out.textures_delta.set {
            r.egui_rend.update_texture(&r.gpu.device, &r.gpu.queue, *id, delta);
        }
        let mut enc = r.gpu.device.create_command_encoder(&Default::default());
        r.view.render(&mut enc, &r.gpu.queue, vp, [0.0037, 0.0044, 0.0051], r.settings.line_px);
        let extra = r.egui_rend.update_buffers(&r.gpu.device, &r.gpu.queue, &mut enc, &jobs, &sd);
        {
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            r.egui_rend.render(&mut rp, &jobs, &sd);
        }
        r.gpu.queue.submit(extra.into_iter().chain(std::iter::once(enc.finish())));
        let present_start = Instant::now();
        frame.present();
        waited += present_start.elapsed();
        // Frames that miss a 120 Hz slot, with where the time went (RUST_LOG=rsviz=debug).
        let total = frame_start.elapsed();
        if log::log_enabled!(log::Level::Debug) && (total + gap).as_secs_f32() > 0.0095 {
            let ms = |d: Duration| d.as_secs_f32() * 1e3;
            log::debug!(
                "slow frame {:.1} ms: idle before {:.1}, ui {:.1}, scene {:.1}, acquire {:.1}, draw {:.1}, present {:.1}",
                ms(total + gap),
                ms(gap),
                ms(t_ui),
                ms(t_scene),
                ms(t_acquire),
                ms(present_start.duration_since(acquire_start).saturating_sub(t_acquire)),
                ms(present_start.elapsed())
            );
        }
        let work = frame_start.elapsed().saturating_sub(waited).as_secs_f32() * 1e3;
        r.frame_ms = if r.frame_ms == 0.0 { work } else { 0.9 * r.frame_ms + 0.1 * work };
        r.wait_ms = 0.9 * r.wait_ms + 0.1 * waited.as_secs_f32() * 1e3;
        static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 100 == 99 {
            log::debug!(
                "{fps:.0} fps, frame {:.1} ms, waiting for the display {:.1} ms, monitor {:.0} Hz, cloud points drawn {:.2} of {:.2} M",
                r.frame_ms,
                r.wait_ms,
                r.refresh,
                r.view.cloud_points.0 as f64 * 1e-6,
                r.view.cloud_points.1 as f64 * 1e-6
            );
        }
        for id in &out.textures_delta.free {
            r.egui_rend.free_texture(id);
        }

        // Continuous drawing is paced by vsync. Where vsync does not hold it back (seen as a frame rate above the
        // refresh rate of the monitor), it is paced by a sleep instead, for good.
        if r.refresh_at.elapsed() >= Duration::from_secs(1) {
            r.refresh_at = Instant::now();
            r.refresh = r.window.current_monitor().and_then(|m| m.refresh_rate_millihertz()).map_or(60.0, |m| m as f32 / 1000.0).max(24.0);
        }
        if !self.args.on_demand && fps > 1.15 * r.refresh && !r.limit {
            log::info!("vsync does not pace this display: capped at {:.0} fps", r.refresh);
            r.limit = true;
        }
        let cap = self.args.max_fps.or((!self.args.on_demand && r.limit).then_some(r.refresh));
        let now = Instant::now();
        r.last_frame = match cap {
            // Paced on a fixed timeline (not "sleep, then restart the clock"), so that the rate does not drift low.
            Some(f) => {
                let next = r.last_frame + Duration::from_secs_f32(1.0 / f);
                if next > now {
                    std::thread::sleep(next - now);
                    next
                } else {
                    now
                }
            }
            None => now,
        };
        let delay = out.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay).unwrap_or(Duration::MAX);
        if !self.args.on_demand || delay.is_zero() {
            if self.args.on_demand && std::env::var("RSVIZ_DEBUG_REPAINT").is_ok() {
                log::warn!("repaint requested by {:?}", r.egui_ctx.repaint_causes());
            }
            r.window.request_redraw();
        }
    }
}

fn to_mat4(a: &DAffine3) -> Mat4 {
    Mat4::from_cols_array(&glam::DMat4::from(*a).to_cols_array().map(|v| v as f32))
}

fn build_lines(s: &SceneData, fixed: &str, visible: &HashMap<String, bool>, styles: &HashMap<String, ui::TopicStyle>, tf: &ui::TfStyle, grid: Option<(Vec3, f32)>, axes: bool) -> Vec<LineVertex> {
    let mut v = Vec::new();
    let mut gv: Vec<LineVertex> = Vec::new();
    // w: width in pixels, negative = the line width of the view
    let mut seg = |a: Vec3, b: Vec3, c: [f32; 4], w: f32| {
        v.push(LineVertex { pos: a.to_array(), width: w, color: c });
        v.push(LineVertex { pos: b.to_array(), width: w, color: c });
    };
    if let Some((c, step)) = grid {
        // 40 x 40 cells, fading out with the distance from the centre (each line split so the fade shows).
        let n = 20;
        let ext = n as f32 * step;
        let base = [0.36, 0.39, 0.43];
        let col = |p: Vec3| {
            let d = ((p - c).truncate().length() / ext).min(1.0);
            [base[0], base[1], base[2], 0.55 * (1.0 - d * d)]
        };
        for i in -n..=n {
            let major = i % 5 == 0;
            for k in -n..n {
                let (a0, a1) = (k as f32 * step, (k + 1) as f32 * step);
                let f = i as f32 * step;
                for (p, q) in [(Vec3::new(f, a0, 0.0), Vec3::new(f, a1, 0.0)), (Vec3::new(a0, f, 0.0), Vec3::new(a1, f, 0.0))] {
                    let (p, q) = (c + p, c + q);
                    let (mut cp, mut cq) = (col(p), col(q));
                    if !major {
                        cp[3] *= 0.5;
                        cq[3] *= 0.5;
                    }
                    gv.push(LineVertex { pos: p.to_array(), width: 1.0, color: cp });
                    gv.push(LineVertex { pos: q.to_array(), width: 1.0, color: cq });
                }
            }
        }
    }
    if axes {
        for f in s.tf.frames() {
            if tf.hidden.contains(&f) {
                continue;
            }
            if let Some(a) = s.tf.lookup(fixed, &f) {
                let o = a.translation.as_vec3();
                let m = a.matrix3.as_mat3();
                seg(o, o + m * Vec3::X * tf.len, [0.9, 0.2, 0.2, 1.0], -1.0);
                seg(o, o + m * Vec3::Y * tf.len, [0.2, 0.85, 0.3, 1.0], -1.0);
                seg(o, o + m * Vec3::Z * tf.len, [0.25, 0.45, 0.95, 1.0], -1.0);
            }
        }
    }
    for (topic, ms) in &s.markers {
        if !visible.get(topic).copied().unwrap_or(true) {
            continue;
        }
        let mw = styles.get(topic).and_then(|t| t.width).unwrap_or(-1.0);
        for d in ms.values() {
            let Prim::Lines(p, c) = &d.prim else { continue };
            let Some(a) = s.tf.lookup(fixed, &d.frame_id) else { continue };
            let a = a * d.pose;
            for (k, w) in p.chunks_exact(2).enumerate() {
                let q0 = a.transform_point3(glam::DVec3::new(w[0][0] as f64, w[0][1] as f64, w[0][2] as f64)).as_vec3();
                let q1 = a.transform_point3(glam::DVec3::new(w[1][0] as f64, w[1][1] as f64, w[1][2] as f64)).as_vec3();
                seg(q0, q1, c[2 * k], mw);
            }
        }
    }
    let mut names: Vec<&String> = s.lines.keys().collect();
    names.sort();
    for (i, name) in names.into_iter().enumerate() {
        if !visible.get(name).copied().unwrap_or(true) {
            continue;
        }
        let l = &s.lines[name];
        let Some(a) = s.tf.lookup(fixed, &l.frame_id) else { continue };
        let st = styles.get(name).copied().unwrap_or_default();
        let c = st.color.map(|c| [c[0], c[1], c[2], 1.0]).unwrap_or(PALETTE[i % PALETTE.len()]);
        let lw = st.width.unwrap_or(-1.0);
        let pts: Vec<Vec3> = l.points.iter().map(|p| a.transform_point3(glam::DVec3::from(*p)).as_vec3()).collect();
        for w in pts.windows(2) {
            seg(w[0], w[1], c, lw);
        }
    }
    v.extend(gv);
    v
}

impl ApplicationHandler<()> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.run.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("rsviz").with_inner_size(winit::dpi::LogicalSize::new(self.args.width, self.args.height)))
                .expect("create window"),
        );
        // Continuous drawing does not wait on the compositor's vsync (on some desktops it runs below the refresh rate
        // of the monitor): frames are presented at once and the loop is paced at the refresh rate by the clock.
        // On demand, plain vsync.
        use wgpu::PresentMode::{Fifo, Immediate, Mailbox};
        let present: Vec<wgpu::PresentMode> = match self.args.present {
            Some(p) => vec![p],
            None if self.args.on_demand => vec![Fifo],
            None => vec![Mailbox, Immediate],
        };
        let gpu = pollster::block_on(Gpu::new(window.clone(), &present, self.args.gpu.as_deref()));
        let limit = gpu.config.present_mode != Fifo;
        if self.args.fullscreen {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(window.primary_monitor())));
        }
        let egui_ctx = egui::Context::default();
        ui::install_theme(&egui_ctx);
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        let mut egui_rend = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, None, 1, false);
        let view = View3D::new(&gpu.device, &gpu.queue, &mut egui_rend);
        self.run = Some(Running {
            window,
            gpu,
            egui_ctx,
            egui_state,
            egui_rend,
            view,
            images: HashMap::new(),
            camera: Camera { dist: self.args.distance, pitch: self.args.pitch, ..Camera::default() },
            visible: HashMap::new(),
            styles: HashMap::new(),
            tf_style: ui::TfStyle { hidden: Default::default(), len: 0.5 },
            style_rev: 0,
            lines_key: (u64::MAX, String::new(), false, false, [0; 3], 0, 0),
            settings: ui::ViewSettings { point_px: self.args.point_px, line_px: self.args.line_px, grid: true, axes: true },
            tree: ui::build_tree(&[]),
            tree_key: Vec::new(),
            timeline: ui::TimelineState::default(),
            framed_bag: 0,
            frame_times: Default::default(),
            frame_ms: 0.0,
            wait_ms: 0.0,
            last_frame: Instant::now(),
            refresh: 60.0,
            refresh_at: Instant::now() - Duration::from_secs(10),
            limit,
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(r) = self.run.as_mut() else { return };
        let resp = r.egui_state.on_window_event(&r.window, &event);
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                r.gpu.resize(s.width, s.height);
                r.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::KeyboardInput { event: winit::event::KeyEvent { physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::F11), state: winit::event::ElementState::Pressed, repeat: false, .. }, .. } => {
                let full = r.window.fullscreen().is_some();
                r.window.set_fullscreen((!full).then_some(winit::window::Fullscreen::Borderless(None)));
            }
            WindowEvent::DroppedFile(path) => {
                r.timeline = ui::TimelineState::default();
                player::open(&self.shared, vec![path], self.args.bag_config());
            }
            _ => {
                if resp.repaint {
                    r.window.request_redraw();
                }
            }
        }
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, _e: ()) {
        if let Some(r) = &self.run {
            r.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn")).init();
    let args = parse_args();
    if args.bag_info {
        std::process::exit(player::info(args.bags.clone()));
    }
    let el = EventLoop::<()>::with_user_event().build().expect("event loop");
    el.set_control_flow(ControlFlow::Wait);
    let shared = Arc::new(Shared::new(el.create_proxy()));
    if !args.bags.is_empty() {
        // Muted before the ROS thread starts: no live topic shows up before the bag.
        shared.live.store(false, std::sync::atomic::Ordering::Release);
        player::open(&shared, args.bags.clone(), args.bag_config());
    }
    ros::spawn(shared.clone(), ros::Options { topics: args.topics.clone(), trail: args.trail });
    let mut app = App { args, shared, run: None };
    el.run_app(&mut app).expect("event loop");
}
