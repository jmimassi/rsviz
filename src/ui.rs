//! Look and layout, in the spirit of the Rerun viewer: a dark, compact theme (Inter), a topic list grouped by kind, and
//! the views as draggable tiles (egui_tiles), one 3D view and one tile per camera, each with its title.

use std::collections::{BTreeMap, HashMap};

use egui::{Color32, FontFamily, FontId, RichText, Stroke, TextStyle};
use glam::Mat4;

use crate::gfx::{Camera, ImageTex};

pub mod pal {
    use egui::Color32;
    pub const BG: Color32 = Color32::from_rgb(0x0c, 0x0e, 0x10);
    pub const PANEL: Color32 = Color32::from_rgb(0x13, 0x15, 0x18);
    pub const HEADER: Color32 = Color32::from_rgb(0x18, 0x1b, 0x1f);
    pub const WIDGET: Color32 = Color32::from_rgb(0x1f, 0x23, 0x27);
    pub const HOVER: Color32 = Color32::from_rgb(0x2a, 0x2f, 0x35);
    pub const ACTIVE: Color32 = Color32::from_rgb(0x33, 0x39, 0x40);
    pub const STROKE: Color32 = Color32::from_rgb(0x26, 0x2a, 0x2f);
    pub const TEXT: Color32 = Color32::from_rgb(0xc9, 0xd1, 0xd9);
    pub const WEAK: Color32 = Color32::from_rgb(0x7c, 0x86, 0x90);
    pub const STRONG: Color32 = Color32::from_rgb(0xf2, 0xf5, 0xf8);
    pub const ACCENT: Color32 = Color32::from_rgb(0x3d, 0x8b, 0xff);
    pub const ERROR: Color32 = Color32::from_rgb(0xe5, 0x5a, 0x4f);
}

pub fn install_theme(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert("inter".into(), egui::FontData::from_static(include_bytes!("../assets/Inter.ttf")));
    fonts.families.get_mut(&FontFamily::Proportional).unwrap().insert(0, "inter".into());
    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Small, FontId::proportional(11.0)),
        (TextStyle::Body, FontId::proportional(13.0)),
        (TextStyle::Button, FontId::proportional(13.0)),
        (TextStyle::Heading, FontId::proportional(15.0)),
        (TextStyle::Monospace, FontId::monospace(12.0)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.button_padding = egui::vec2(7.0, 3.0);
    style.spacing.interact_size.y = 20.0;
    style.spacing.combo_width = 140.0;
    let v = &mut style.visuals;
    *v = egui::Visuals::dark();
    v.panel_fill = pal::PANEL;
    v.window_fill = pal::PANEL;
    v.extreme_bg_color = pal::BG;
    v.faint_bg_color = pal::HEADER;
    v.override_text_color = Some(pal::TEXT);
    v.hyperlink_color = pal::ACCENT;
    v.selection.bg_fill = pal::ACCENT.gamma_multiply(0.45);
    v.selection.stroke = Stroke::new(1.0f32, pal::ACCENT);
    v.window_stroke = Stroke::new(1.0f32, pal::STROKE);
    v.window_rounding = egui::Rounding::same(6.0);
    v.menu_rounding = egui::Rounding::same(6.0);
    v.widgets.noninteractive.bg_fill = pal::PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0f32, pal::STROKE);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0f32, pal::TEXT);
    for (w, fill) in [(&mut v.widgets.inactive, pal::WIDGET), (&mut v.widgets.hovered, pal::HOVER), (&mut v.widgets.active, pal::ACTIVE), (&mut v.widgets.open, pal::HOVER)] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.bg_stroke = Stroke::NONE;
        w.rounding = egui::Rounding::same(4.0);
        w.expansion = 0.0;
    }
    v.widgets.inactive.fg_stroke = Stroke::new(1.0f32, pal::TEXT);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0f32, pal::STRONG);
    v.widgets.active.fg_stroke = Stroke::new(1.0f32, pal::STRONG);
    ctx.set_style(style);
}

pub struct TopicRow {
    pub name: String,
    pub kind: &'static str,
    pub hz: f32,
    pub qos: String,
    pub error: Option<String>,
}

/// What the user set for one topic in the topics panel; None = the default of the view.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct TopicStyle {
    /// Lines (paths, trails, marker lines): width in pixels.
    pub width: Option<f32>,
    /// Paths and trails: colour.
    pub color: Option<[f32; 3]>,
    /// Point clouds: point size in pixels.
    pub point_px: Option<f32>,
}

/// The TF frames: which ones are not drawn, and the length of their axes (m).
pub struct TfStyle {
    pub hidden: std::collections::HashSet<String>,
    pub len: f32,
}

/// Which settings a topic of this kind has: (line width, colour, point size).
fn style_of(kind: &str) -> (bool, bool, bool) {
    match kind {
        "Path" | "Odometry" | "PoseStamped" => (true, true, false),
        "Marker" | "MarkerArray" => (true, false, false),
        "PointCloud2" => (false, false, true),
        _ => (false, false, false),
    }
}

fn kind_color(kind: &str) -> Color32 {
    match kind {
        "PointCloud2" => Color32::from_rgb(0xee, 0xa1, 0x2c),
        "Image" | "CompressedImage" => Color32::from_rgb(0x2d, 0xb3, 0x7f),
        "Path" | "Odometry" | "PoseStamped" | "PointStamped" => Color32::from_rgb(0x3d, 0x8b, 0xff),
        "Marker" | "MarkerArray" => Color32::from_rgb(0xb0, 0x7c, 0xf0),
        "TF" | "RobotModel" => Color32::from_rgb(0x8f, 0x99, 0xa3),
        _ => pal::WEAK,
    }
}

fn is_image(kind: &str) -> bool {
    matches!(kind, "Image" | "CompressedImage")
}

/// A small eye-like toggle: filled when shown, hollow when hidden.
fn eye(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
    }
    let c = rect.center();
    let col = if resp.hovered() { pal::STRONG } else if *on { pal::TEXT } else { pal::WEAK };
    ui.painter().circle_stroke(c, 5.5, Stroke::new(1.3f32, col));
    if *on {
        ui.painter().circle_filled(c, 2.6, col);
    }
    resp.on_hover_text(if *on { "Hide" } else { "Show" })
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title.to_uppercase()).size(10.5).color(pal::WEAK).strong());
    ui.add_space(2.0);
}

/// The topics, with their settings (click on a name). defaults: (line width, point size) of the view. Returns true when
/// a setting changed.
pub fn topics_panel(ctx: &egui::Context, rows: &[TopicRow], visible: &mut HashMap<String, bool>, styles: &mut HashMap<String, TopicStyle>, defaults: (f32, f32), frames: &[String], tf: &mut TfStyle) -> bool {
    let mut changed = false;
    egui::SidePanel::left("topics")
        .default_width(270.0)
        .frame(egui::Frame::none().fill(pal::PANEL).inner_margin(egui::Margin::symmetric(10.0, 8.0)))
        .show(ctx, |ui| {
            ui.label(RichText::new("Topics").heading().color(pal::STRONG));
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                if rows.is_empty() {
                    ui.add_space(8.0);
                    ui.label(RichText::new("Waiting for publishers: PointCloud2, Image, CompressedImage, Path, Odometry, PoseStamped, PointStamped, Marker(Array), TF, robot_description.\n\nOr open a bag (ROS 1 or ROS 2): Open bag, or drop it on the window.").color(pal::WEAK));
                }
                for (title, images) in [("3D", false), ("Images", true)] {
                    let group: Vec<&TopicRow> = rows.iter().filter(|r| is_image(r.kind) == images).collect();
                    if group.is_empty() {
                        continue;
                    }
                    section(ui, title);
                    for row in group {
                        let (has_w, has_c, has_p) = style_of(row.kind);
                        let is_tf = row.kind == "TF";
                        let has_style = has_w || has_c || has_p || is_tf;
                        let open_id = egui::Id::new(("topic settings", &row.name));
                        let mut open = ui.data(|d| d.get_temp::<bool>(open_id)).unwrap_or(false);
                        let mut on = visible.get(&row.name).copied().unwrap_or(true);
                        let resp = ui
                            .horizontal(|ui| {
                                if eye(ui, &mut on).clicked() {
                                    visible.insert(row.name.clone(), on);
                                }
                                let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 16.0), egui::Sense::hover());
                                let dc = if row.error.is_some() { pal::ERROR } else { kind_color(row.kind) };
                                ui.painter().rect_filled(egui::Rect::from_center_size(dot.center(), egui::vec2(7.0, 7.0)), 2.0f32, dc);
                                let name_col = if on { pal::TEXT } else { pal::WEAK };
                                let hz = if row.hz >= 10.0 { format!("{:.0} Hz", row.hz) } else { format!("{:.1} Hz", row.hz) };
                                let name_w = (ui.available_width() - 46.0).max(20.0);
                                ui.allocate_ui_with_layout(egui::vec2(name_w, 18.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                    ui.set_min_width(name_w);
                                    let shown = if has_style { format!("{} {}", row.name, if open { "⏷" } else { "⏵" }) } else { row.name.clone() };
                                    let name = ui.add(egui::Label::new(RichText::new(shown).color(name_col)).truncate().sense(egui::Sense::click()));
                                    if has_style && name.clicked() {
                                        open = !open;
                                        ui.data_mut(|d| d.insert_temp(open_id, open));
                                    }
                                });
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(RichText::new(hz).monospace().size(11.0).color(pal::WEAK));
                                });
                            })
                            .response;
                        let more = if has_style { "\nclick on the name: settings" } else { "" };
                        let tip = format!("{}\n{} · {}{}{more}", row.name, row.kind, row.qos, row.error.as_ref().map(|e| format!("\n{e}")).unwrap_or_default());
                        resp.on_hover_text(tip);
                        if is_tf && open {
                            // the frames, one by one: which are drawn, and how long their axes are
                            ui.horizontal(|ui| {
                                ui.add_space(30.0);
                                if ui.add(egui::DragValue::new(&mut tf.len).range(0.05..=1000.0).speed(0.05).suffix(" m")).on_hover_text("length of the axes").changed() {
                                    changed = true;
                                }
                                if ui.small_button("all").clicked() {
                                    tf.hidden.clear();
                                    changed = true;
                                }
                                if ui.small_button("none").clicked() {
                                    tf.hidden = frames.iter().cloned().collect();
                                    changed = true;
                                }
                            });
                            for f in frames {
                                ui.horizontal(|ui| {
                                    ui.add_space(30.0);
                                    let mut on = !tf.hidden.contains(f);
                                    if eye(ui, &mut on).clicked() {
                                        if on {
                                            tf.hidden.remove(f);
                                        } else {
                                            tf.hidden.insert(f.clone());
                                        }
                                        changed = true;
                                    }
                                    ui.add(egui::Label::new(RichText::new(f).size(12.0).color(if on { pal::TEXT } else { pal::WEAK })).truncate());
                                });
                            }
                        } else if has_style && open {
                            let old = styles.get(&row.name).copied().unwrap_or_default();
                            let mut st = old;
                            ui.horizontal(|ui| {
                                ui.add_space(30.0);
                                ui.spacing_mut().slider_width = 86.0;
                                if has_w {
                                    let mut w = st.width.unwrap_or(defaults.0);
                                    if ui.add(egui::Slider::new(&mut w, 1.0..=10.0).fixed_decimals(1)).on_hover_text("line width (px)").changed() {
                                        st.width = Some(w);
                                    }
                                }
                                if has_p {
                                    let mut p = st.point_px.unwrap_or(defaults.1);
                                    if ui.add(egui::Slider::new(&mut p, 1.0..=10.0).fixed_decimals(1)).on_hover_text("point size (px)").changed() {
                                        st.point_px = Some(p);
                                    }
                                }
                                if has_c {
                                    let mut c = st.color.unwrap_or([0.24, 0.55, 1.0]);
                                    if ui.color_edit_button_rgb(&mut c).on_hover_text("colour").changed() {
                                        st.color = Some(c);
                                    }
                                }
                                if st != TopicStyle::default() && ui.small_button("reset").on_hover_text("back to the defaults of the view").clicked() {
                                    st = TopicStyle::default();
                                }
                            });
                            if st != old {
                                styles.insert(row.name.clone(), st);
                                changed = true;
                            }
                        }
                    }
                }
            });
        });
    changed
}

pub struct TopBar<'a> {
    pub fixed_frame: &'a mut String,
    pub follow: &'a mut Option<String>,
    pub frames: &'a [String],
    /// Frames drawn per second, and what one frame costs (ms, without the wait for the display).
    pub fps: f32,
    pub frame_ms: f32,
    /// Time per frame spent waiting for the display, ms, and the refresh rate of the monitor (Hz).
    pub wait_ms: f32,
    pub refresh: f32,
    /// Frames are drawn only when something changes (--on-demand), not at the display rate.
    pub on_demand: bool,
    pub n_topics: usize,
    /// Where the topics come from: "ROS domain 0", or the name of the open bag.
    pub source: String,
    /// Live ROS domain, None while a bag is open.
    pub domain: Option<u32>,
    /// "Opening x.mcap" while a bag is being indexed.
    pub loading: Option<String>,
    /// Last bag error or warning.
    pub notice: Option<String>,
}

#[derive(Default)]
pub struct TopAction {
    /// Focus was clicked.
    pub focus: bool,
    /// "Open bag" was clicked.
    pub open: bool,
    /// The notice was dismissed.
    pub dismiss: bool,
    /// Another ROS domain was picked.
    pub domain: Option<u32>,
}

pub fn top_bar(ctx: &egui::Context, b: TopBar) -> TopAction {
    let mut act = TopAction::default();
    egui::TopBottomPanel::top("top")
        .exact_height(34.0)
        .frame(egui::Frame::none().fill(pal::HEADER).inner_margin(egui::Margin::symmetric(10.0, 5.0)).stroke(Stroke::new(1.0f32, pal::STROKE)))
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 4.5, pal::ACCENT);
                ui.label(RichText::new("rsviz").strong().color(pal::STRONG));
                ui.add_space(14.0);
                ui.label(RichText::new("Fixed frame").color(pal::WEAK));
                egui::ComboBox::from_id_salt("ff").selected_text(b.fixed_frame.clone()).show_ui(ui, |ui| {
                    for f in b.frames {
                        ui.selectable_value(b.fixed_frame, f.clone(), f);
                    }
                });
                ui.add_space(8.0);
                ui.label(RichText::new("Follow").color(pal::WEAK));
                let shown = match b.follow { Some(f) if !f.is_empty() => f.clone(), Some(_) => "auto".into(), None => "none".into() };
                egui::ComboBox::from_id_salt("follow").selected_text(shown).show_ui(ui, |ui| {
                    ui.selectable_value(b.follow, Some(String::new()), "auto (base_link)");
                    ui.selectable_value(b.follow, None, "none");
                    for f in b.frames {
                        ui.selectable_value(b.follow, Some(f.clone()), f);
                    }
                });
                if ui.button("Focus").on_hover_text("Recentre on the followed frame, or on the scene").clicked() {
                    act.focus = true;
                }
                ui.add_space(8.0);
                if ui.button("Open bag").on_hover_text("Play a ROS 1 bag (.bag) or a ROS 2 bag (.mcap, .db3, or its directory).\nA bag can also be dropped on the window.").clicked() {
                    act.open = true;
                }
                if let Some(l) = &b.loading {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.label(RichText::new(l).color(pal::TEXT));
                } else if let Some(n) = &b.notice {
                    // Room is kept for the status on the right; the full text is in the tooltip.
                    let w = (ui.available_width() - 440.0).max(60.0);
                    ui.allocate_ui_with_layout(egui::vec2(w, 20.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        if ui.add(egui::Label::new(RichText::new(n).color(pal::ERROR)).truncate().sense(egui::Sense::click())).on_hover_text(format!("{n}\n(click to dismiss)")).clicked() {
                            act.dismiss = true;
                        }
                    });
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (unit, tip) = if b.on_demand {
                        ("redraws/s", "On demand: a frame is drawn only when something changes (a new message, the bag cursor, the mouse), so this rate follows the data.")
                    } else {
                        ("fps", "Frames drawn per second, paced by the refresh rate of the display (rsviz --on-demand draws only when something changes).")
                    };
                    ui.label(RichText::new(format!("{:.0} {unit} · {:.1} ms", b.fps, b.frame_ms)).monospace().color(pal::WEAK))
                        .on_hover_text(format!(
                            "{tip}\nThe time is what one frame costs here. Waiting for the display: {:.1} ms per frame. Monitor: {:.0} Hz.\nA rate under the monitor's with a short frame time and a long wait is the desktop (compositor, GPU that drives the screens) not taking frames faster.",
                            b.wait_ms, b.refresh
                        ));
                    ui.label(RichText::new("·").color(pal::WEAK));
                    ui.label(RichText::new(format!("{} topics", b.n_topics)).color(pal::WEAK));
                    ui.label(RichText::new("·").color(pal::WEAK));
                    match b.domain {
                        Some(cur) => {
                            ui.menu_button(RichText::new(format!("ROS domain {cur} ▾")).color(pal::WEAK), |ui| {
                                let id = egui::Id::new("domain_edit");
                                let mut v: u32 = ui.data_mut(|d| *d.get_temp_mut_or(id, cur));
                                ui.horizontal(|ui| {
                                    ui.label("Domain");
                                    let r = ui.add(egui::DragValue::new(&mut v).range(0..=232));
                                    let go = ui.button("Connect").clicked() || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                                    if go && v != cur {
                                        act.domain = Some(v);
                                        ui.close_menu();
                                    }
                                });
                                ui.data_mut(|d| d.insert_temp(id, v));
                                ui.label(RichText::new("Same as ROS_DOMAIN_ID of the publishers.").color(pal::WEAK));
                            })
                            .response
                            .on_hover_text("Live topics come from this ROS domain. Click to listen on another one.");
                        }
                        None => {
                            ui.add(egui::Label::new(RichText::new(&b.source).color(pal::WEAK)).truncate());
                        }
                    }
                });
            });
        });
    act
}

#[derive(Clone, PartialEq)]
pub enum Pane {
    View3D,
    Image(String),
}

/// Default layout: the 3D view on the left, the cameras stacked on the right.
pub fn build_tree(images: &[String]) -> egui_tiles::Tree<Pane> {
    let mut tiles = egui_tiles::Tiles::default();
    let v = tiles.insert_pane(Pane::View3D);
    let root = if images.is_empty() {
        tiles.insert_tab_tile(vec![v])
    } else {
        let ims: Vec<_> = images.iter().map(|t| tiles.insert_pane(Pane::Image(t.clone()))).collect();
        let col = tiles.insert_vertical_tile(ims);
        tiles.insert_container(egui_tiles::Linear::new_binary(egui_tiles::LinearDir::Horizontal, [v, col], 0.68))
    };
    egui_tiles::Tree::new("views", root, tiles)
}

pub struct ViewSettings {
    pub point_px: f32,
    pub line_px: f32,
    pub grid: bool,
    pub axes: bool,
}

pub struct Behavior<'a> {
    pub view_tex: egui::TextureId,
    pub view_px: &'a mut (u32, u32),
    pub camera: &'a mut Camera,
    pub vp: Mat4,
    pub texts: &'a [(glam::Vec3, String, [f32; 4])],
    pub images: &'a HashMap<String, Option<ImageTex>>,
    pub image_meta: &'a BTreeMap<String, (u32, u32, f32)>,
    pub hz: &'a HashMap<String, f32>,
    pub settings: &'a mut ViewSettings,
}

impl egui_tiles::Behavior<Pane> for Behavior<'_> {
    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        match pane {
            Pane::View3D => "3D".into(),
            Pane::Image(t) => t.as_str().into(),
        }
    }

    fn pane_ui(&mut self, ui: &mut egui::Ui, _id: egui_tiles::TileId, pane: &mut Pane) -> egui_tiles::UiResponse {
        let rect = ui.available_rect_before_wrap();
        ui.painter().rect_filled(rect, 0.0f32, pal::BG);
        match pane {
            Pane::View3D => {
                let ppp = ui.ctx().pixels_per_point();
                *self.view_px = ((rect.width() * ppp) as u32, (rect.height() * ppp) as u32);
                let resp = ui.put(rect, egui::Image::new(egui::load::SizedTexture::new(self.view_tex, rect.size())).sense(egui::Sense::click_and_drag()));
                let painter = ui.painter_at(rect);
                for (w, text, c) in self.texts {
                    let clip = self.vp * w.extend(1.0);
                    if clip.w <= 0.0 {
                        continue;
                    }
                    let ndc = clip.truncate() / clip.w;
                    let px = rect.min + egui::vec2((ndc.x + 1.0) * 0.5 * rect.width(), (1.0 - ndc.y) * 0.5 * rect.height());
                    let col = Color32::from_rgba_unmultiplied((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8, (c[3] * 255.0) as u8);
                    painter.text(px + egui::vec2(1.0, 1.0), egui::Align2::CENTER_CENTER, text, FontId::proportional(13.0), Color32::from_black_alpha(160));
                    painter.text(px, egui::Align2::CENTER_CENTER, text, FontId::proportional(13.0), col);
                }
                let d = resp.drag_delta();
                if resp.dragged_by(egui::PointerButton::Primary) {
                    self.camera.orbit(d.x, d.y);
                } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
                    self.camera.pan(d.x, d.y, rect.height());
                }
                if resp.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.zoom(scroll / 50.0);
                    }
                }
                // View options, top-right of the view.
                let opt = egui::Rect::from_min_size(egui::pos2(rect.right() - 92.0, rect.top() + 8.0), egui::vec2(84.0, 22.0));
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(opt), |ui| {
                    ui.menu_button("View ⏷", |ui| {
                        ui.add(egui::Slider::new(&mut self.settings.point_px, 1.0..=10.0).text("point size"));
                        ui.add(egui::Slider::new(&mut self.settings.line_px, 1.0..=10.0).text("line width"));
                        ui.checkbox(&mut self.settings.grid, "Grid");
                        ui.checkbox(&mut self.settings.axes, "TF axes");
                    });
                });
                painter.text(
                    rect.left_bottom() + egui::vec2(10.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    "left drag: orbit · right drag: pan · wheel: zoom",
                    FontId::proportional(11.0),
                    pal::WEAK,
                );
            }
            Pane::Image(topic) => {
                if let (Some(Some(tex)), Some((w, h, ms))) = (self.images.get(topic.as_str()), self.image_meta.get(topic.as_str())) {
                    let aspect = *w as f32 / (*h).max(1) as f32;
                    let mut size = rect.size();
                    if size.x / size.y.max(1.0) > aspect {
                        size.x = size.y * aspect;
                    } else {
                        size.y = size.x / aspect;
                    }
                    let r = egui::Rect::from_center_size(rect.center(), size);
                    ui.painter().image(tex.id, r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
                    let hz = self.hz.get(topic.as_str()).copied().unwrap_or(0.0);
                    let label = format!("{w}×{h} · {hz:.1} Hz · decode {ms:.1} ms");
                    let g = ui.painter().layout_no_wrap(label, FontId::proportional(11.0), pal::TEXT);
                    let bg = egui::Rect::from_min_size(rect.left_bottom() + egui::vec2(8.0, -8.0 - g.size().y - 6.0), g.size() + egui::vec2(10.0, 6.0));
                    ui.painter().rect_filled(bg, 4.0f32, Color32::from_black_alpha(170));
                    ui.painter().galley(bg.min + egui::vec2(5.0, 3.0), g, pal::TEXT);
                } else {
                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "waiting for images", FontId::proportional(12.0), pal::WEAK);
                }
                ui.allocate_rect(rect, egui::Sense::hover());
            }
        }
        egui_tiles::UiResponse::None
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions { all_panes_must_have_tabs: true, ..Default::default() }
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        26.0
    }

    fn gap_width(&self, _style: &egui::Style) -> f32 {
        2.0
    }

    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        pal::HEADER
    }

    fn tab_bg_color(&self, _visuals: &egui::Visuals, _tiles: &egui_tiles::Tiles<Pane>, _tile_id: egui_tiles::TileId, state: &egui_tiles::TabState) -> Color32 {
        if state.active { pal::PANEL } else { pal::HEADER }
    }

    fn tab_text_color(&self, _visuals: &egui::Visuals, _tiles: &egui_tiles::Tiles<Pane>, _tile_id: egui_tiles::TileId, state: &egui_tiles::TabState) -> Color32 {
        if state.active { pal::STRONG } else { pal::WEAK }
    }

    fn tab_outline_stroke(&self, _visuals: &egui::Visuals, _tiles: &egui_tiles::Tiles<Pane>, _tile_id: egui_tiles::TileId, _state: &egui_tiles::TabState) -> Stroke {
        Stroke::NONE
    }

    fn tab_bar_hline_stroke(&self, _visuals: &egui::Visuals) -> Stroke {
        Stroke::new(1.0f32, pal::STROKE)
    }

    fn resize_stroke(&self, _style: &egui::Style, state: egui_tiles::ResizeState) -> Stroke {
        match state {
            egui_tiles::ResizeState::Idle => Stroke::new(1.0f32, pal::STROKE),
            _ => Stroke::new(2.0f32, pal::ACCENT),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Bag timeline: transport controls, a time axis and one lane per topic (message density), with the play cursor.
// ---------------------------------------------------------------------------------------------------------------

#[derive(Default)]
pub struct TimelineState {
    /// Visible part of the bag in seconds from its start (None = the whole bag).
    pub view: Option<(f64, f64)>,
    /// Lanes for every topic of the bag, not only the ones rsviz shows.
    pub all_topics: bool,
    /// Last time asked by the pointer (a still pointer does not seek again).
    last_seek: Option<u64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Icon {
    Play,
    Pause,
    Start,
    StepBack,
    StepFwd,
    Loop,
    Close,
}

fn icon_button(ui: &mut egui::Ui, icon: Icon, on: bool, tip: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(26.0, 22.0), egui::Sense::click());
    let bg = if resp.is_pointer_button_down_on() {
        pal::ACTIVE
    } else if resp.hovered() {
        pal::HOVER
    } else if on {
        pal::ACCENT.gamma_multiply(0.35)
    } else {
        pal::WIDGET
    };
    let p = ui.painter();
    p.rect_filled(rect, 4.0f32, bg);
    let col = if resp.hovered() || on { pal::STRONG } else { pal::TEXT };
    let c = rect.center();
    // Triangle of half-height h centred on cx, pointing right (dir = 1) or left (dir = -1); vertical bar of half-height h.
    let tri = |cx: f32, dir: f32, h: f32| {
        let pts = vec![egui::pos2(cx - dir * h * 0.8, c.y - h), egui::pos2(cx + dir * h * 0.8, c.y), egui::pos2(cx - dir * h * 0.8, c.y + h)];
        p.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
    };
    let bar = |cx: f32, h: f32| p.rect_filled(egui::Rect::from_center_size(egui::pos2(cx, c.y), egui::vec2(2.0, 2.0 * h)), 0.0f32, col);
    match icon {
        Icon::Play => tri(c.x + 0.5, 1.0, 5.5),
        Icon::Pause => {
            bar(c.x - 3.0, 5.0);
            bar(c.x + 3.0, 5.0);
        }
        Icon::Start => {
            bar(c.x - 5.0, 4.5);
            tri(c.x + 1.0, -1.0, 4.5);
        }
        Icon::StepBack => {
            tri(c.x - 1.5, -1.0, 4.5);
            bar(c.x + 4.0, 4.5);
        }
        Icon::StepFwd => {
            bar(c.x - 4.0, 4.5);
            tri(c.x + 1.5, 1.0, 4.5);
        }
        Icon::Loop => {
            // Open circle with an arrow head at its end.
            let r = 5.0;
            let pts: Vec<egui::Pos2> = (0..=20).map(|i| (50.0f32 + i as f32 * 13.5).to_radians()).map(|a| egui::pos2(c.x + r * a.cos(), c.y - r * a.sin())).collect();
            let tip = *pts.last().unwrap();
            p.add(egui::Shape::line(pts, Stroke::new(1.5f32, col)));
            p.add(egui::Shape::convex_polygon(vec![tip + egui::vec2(-3.2, -2.6), tip + egui::vec2(2.8, -0.6), tip + egui::vec2(-1.0, 3.0)], col, Stroke::NONE));
        }
        Icon::Close => {
            let d = 4.0;
            p.line_segment([c + egui::vec2(-d, -d), c + egui::vec2(d, d)], Stroke::new(1.5f32, col));
            p.line_segment([c + egui::vec2(-d, d), c + egui::vec2(d, -d)], Stroke::new(1.5f32, col));
        }
    }
    resp.on_hover_text(tip)
}

/// "mm:ss.mmm" (hours in front for a bag longer than one hour).
fn clock(s: f64, dur: f64) -> String {
    let ms = (s.max(0.0) * 1e3).round() as u64;
    let (h, m, sec, ms) = (ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000);
    if dur >= 3600.0 { format!("{h}:{m:02}:{sec:02}.{ms:03}") } else { format!("{:02}:{sec:02}.{ms:03}", h * 60 + m) }
}

/// Label of an axis tick, with the decimals the tick step needs.
fn tick_label(s: f64, step: f64) -> String {
    let dec = if step >= 1.0 { 0 } else { (-step.log10()).ceil() as usize };
    let total = s.max(0.0);
    let (m, sec) = ((total / 60.0).floor() as u64, total % 60.0);
    let w = if dec == 0 { 2 } else { 3 + dec };
    if m >= 60 { format!("{}:{:02}:{sec:0w$.dec$}", m / 60, m % 60) } else { format!("{m}:{sec:0w$.dec$}") }
}

/// Wall-clock date of a bag time (UTC), or the plain seconds for a clock that does not start at the Unix epoch.
fn utc(ns: u64) -> String {
    let secs = ns / 1_000_000_000;
    if secs < 946_684_800 {
        return format!("t = {:.3} s", ns as f64 * 1e-9);
    }
    let (days, rem, ms) = ((secs / 86_400) as i64, secs % 86_400, ns / 1_000_000 % 1000);
    // Days since the epoch to a civil date (Gregorian calendar).
    let z = days + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}.{ms:03} UTC", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn set_view(view: &mut Option<(f64, f64)>, a: f64, b: f64, dur: f64) {
    let span = (b - a).clamp((dur * 1e-5).max(0.02).min(dur), dur);
    if span >= dur {
        *view = None;
    } else {
        let a = a.clamp(0.0, dur - span);
        *view = Some((a, a + span));
    }
}

/// Geometry of the time tracks: screen x of the lanes and the bag they show.
struct Track<'a> {
    pb: &'a crate::player::Playback,
    dur: f64,
    x0: f32,
    x1: f32,
}

impl Track<'_> {
    /// Pointer on a track (the axis or a lane): left = move the cursor, right / middle drag = pan, Ctrl + wheel or pinch
    /// = zoom around the pointer, horizontal wheel = pan, double click = whole bag.
    fn input(&self, ui: &egui::Ui, resp: &egui::Response, st: &mut TimelineState) {
        let (a, b) = st.view.unwrap_or((0.0, self.dur));
        let w = (self.x1 - self.x0).max(1.0);
        let to_s = |x: f32| a + ((x - self.x0) / w) as f64 * (b - a);
        let primary = ui.input(|i| i.pointer.primary_down());
        if let Some(p) = resp.interact_pointer_pos().filter(|_| (resp.is_pointer_button_down_on() && primary) || resp.clicked()) {
            let t = self.pb.start + (to_s(p.x).clamp(0.0, self.dur) * 1e9) as u64;
            if st.last_seek != Some(t) {
                st.last_seek = Some(t);
                self.pb.seek(t);
            }
        } else if resp.drag_stopped() {
            st.last_seek = None;
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let d = -(resp.drag_delta().x / w) as f64 * (b - a);
            set_view(&mut st.view, a + d, b + d, self.dur);
        }
        if resp.double_clicked() {
            st.view = None;
        }
        if let Some(p) = resp.hover_pos() {
            let (zoom, scroll) = ui.input(|i| (i.zoom_delta() as f64, i.smooth_scroll_delta.x));
            if (zoom - 1.0).abs() > 1e-6 {
                let s = to_s(p.x);
                set_view(&mut st.view, s - (s - a) / zoom, s + (b - s) / zoom, self.dur);
            } else if scroll != 0.0 {
                let d = -(scroll / w) as f64 * (b - a);
                set_view(&mut st.view, a + d, b + d, self.dur);
            }
        }
    }

    /// Message density of one topic over the visible span: one mark per 2 px column that holds messages, dimmed where
    /// the topic is clearly slower than usual (under 40 % of its median rate in the visible span).
    fn density(&self, p: &egui::Painter, lane: egui::Rect, times: &[u64], view: (f64, f64), color: Color32) {
        let (a, b) = view;
        let cols = ((lane.width() / 2.0).ceil() as usize).max(1);
        let at = |c: usize| self.pb.start + ((a + (b - a) * c as f64 / cols as f64).max(0.0) * 1e9) as u64;
        let mut counts = Vec::with_capacity(cols);
        let mut lo = times.partition_point(|t| *t < at(0));
        for c in 0..cols {
            let hi = lo + times[lo..].partition_point(|t| *t < at(c + 1));
            // The very last message of the bag sits on the right edge of the whole view.
            counts.push(hi - lo + usize::from(c + 1 == cols && hi < times.len() && times[hi] == at(cols)));
            lo = hi;
        }
        let mut filled: Vec<usize> = counts.iter().copied().filter(|n| *n > 0).collect();
        if filled.is_empty() {
            return;
        }
        let mid = filled.len() / 2;
        let median = *filled.select_nth_unstable(mid).1;
        let level = |n: usize| if n == 0 { 0u8 } else if (n as f32) < 0.4 * median as f32 { 1 } else { 2 };
        // Runs of columns with the same level are drawn as one rectangle.
        let x = |c: usize| lane.left() + lane.width() * c as f32 / cols as f32;
        let (mut run_start, mut run_level) = (0usize, 0u8);
        for c in 0..=cols {
            let lv = if c < cols { level(counts[c]) } else { 0 };
            if lv != run_level || c == cols {
                if run_level > 0 {
                    let r = egui::Rect::from_min_max(egui::pos2(x(run_start), lane.top()), egui::pos2(x(c).max(x(run_start) + 1.5), lane.bottom()));
                    p.rect_filled(r, 0.0f32, color.gamma_multiply(if run_level == 1 { 0.45 } else { 1.0 }));
                }
                run_start = c;
                run_level = lv;
            }
        }
    }
}

/// The timeline panel of an open bag. Returns true when the user closes the bag.
pub fn timeline(ctx: &egui::Context, pb: &crate::player::Playback, st: &mut TimelineState) -> bool {
    let (playing, speed, looping, cur) = {
        let c = pb.ctl.lock().unwrap();
        // Between two updates by the player the cursor is carried forward, so that it moves at every frame.
        let cur = match c.seek {
            Some(t) => t,
            None if c.playing => c.cur.saturating_add((c.cur_at.elapsed().as_secs_f64().min(0.05) * c.speed * 1e9) as u64).min(pb.end),
            None => c.cur,
        };
        (c.playing, c.speed, c.looping, cur)
    };
    let dur = (pb.end.saturating_sub(pb.start) as f64 * 1e-9).max(1e-6);
    let cur_s = cur.saturating_sub(pb.start) as f64 * 1e-9;
    let mut close = false;

    if !ctx.wants_keyboard_input() {
        use egui::Key;
        let (space, left, right, home, end, shift) = ctx.input(|i| (i.key_pressed(Key::Space), i.key_pressed(Key::ArrowLeft), i.key_pressed(Key::ArrowRight), i.key_pressed(Key::Home), i.key_pressed(Key::End), i.modifiers.shift));
        let step = if shift { 1.0 } else { 0.1 };
        if space {
            pb.toggle();
        }
        if left {
            pb.step(-step);
        }
        if right {
            pb.step(step);
        }
        if home {
            pb.seek(pb.start);
        }
        if end {
            pb.seek(pb.end);
        }
    }

    // Shown topics first, then (on request) the rest of the bag, each group by name.
    let mut rows: Vec<&crate::player::TopicInfo> = pb.topics.iter().filter(|t| st.all_topics || t.kind.is_some()).collect();
    rows.sort_by(|x, y| (x.kind.is_none(), &x.name).cmp(&(y.kind.is_none(), &y.name)));
    let row_h = 18.0;
    let default_h = 34.0 + 24.0 + rows.len().clamp(1, 8) as f32 * (row_h + 2.0) + 12.0;

    egui::TopBottomPanel::bottom("timeline")
        .resizable(true)
        .default_height(default_h)
        .min_height(72.0)
        .frame(egui::Frame::none().fill(pal::PANEL).inner_margin(egui::Margin { left: 10.0, right: 10.0, top: 6.0, bottom: 6.0 }).stroke(Stroke::new(1.0f32, pal::STROKE)))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                if icon_button(ui, Icon::Start, false, "Back to the start (Home)").clicked() {
                    pb.seek(pb.start);
                }
                if icon_button(ui, Icon::StepBack, false, "0.1 s back (Left, with Shift: 1 s)").clicked() {
                    pb.step(-0.1);
                }
                if icon_button(ui, if playing { Icon::Pause } else { Icon::Play }, false, "Play / pause (Space)").clicked() {
                    pb.toggle();
                }
                if icon_button(ui, Icon::StepFwd, false, "0.1 s forward (Right, with Shift: 1 s)").clicked() {
                    pb.step(0.1);
                }
                if icon_button(ui, Icon::Loop, looping, "Loop").clicked() {
                    pb.set_looping(!looping);
                }
                let label = |v: f64| if v < 1.0 { format!("{v}×") } else { format!("{v:.0}×") };
                egui::ComboBox::from_id_salt("speed").width(58.0).selected_text(label(speed)).show_ui(ui, |ui| {
                    for v in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
                        if ui.selectable_label(speed == v, label(v)).clicked() {
                            pb.set_speed(v);
                        }
                    }
                });
                ui.add_space(8.0);
                ui.label(RichText::new(clock(cur_s, dur)).monospace().size(13.0).color(pal::STRONG));
                ui.label(RichText::new(format!("/ {}", clock(dur, dur))).monospace().color(pal::WEAK));
                ui.add_space(8.0);
                ui.label(RichText::new(utc(cur)).monospace().color(pal::WEAK));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if icon_button(ui, Icon::Close, false, "Close the bag and go back to the live topics").clicked() {
                        close = true;
                    }
                    ui.add_space(4.0);
                    if ui.selectable_label(st.all_topics, "all topics").on_hover_text("One lane for every topic of the bag, also the ones rsviz does not show").clicked() {
                        st.all_topics = !st.all_topics;
                    }
                    ui.add_space(4.0);
                    ui.add(egui::Label::new(RichText::new(format!("{} · {} · {} messages", pb.name, pb.format, pb.messages)).color(pal::WEAK)).truncate());
                });
            });
            ui.add_space(2.0);

            let avail = ui.available_rect_before_wrap();
            let name_w = (avail.width() * 0.3).min(250.0);
            let track = Track { pb, dur, x0: avail.left() + name_w, x1: avail.right() - 12.0 };
            // While playing, a zoomed view pages along with the cursor.
            if let Some((a, b)) = st.view {
                if playing && (cur_s > b || cur_s < a) {
                    set_view(&mut st.view, cur_s - 0.05 * (b - a), cur_s + 0.95 * (b - a), dur);
                }
            }
            let view = st.view.unwrap_or((0.0, dur));
            let w = (track.x1 - track.x0).max(1.0);
            let to_x = |s: f64| track.x0 + ((s - view.0) / (view.1 - view.0)) as f32 * w;

            // Time axis.
            let (axis, axis_resp) = ui.allocate_exact_size(egui::vec2(avail.width(), 20.0), egui::Sense::click_and_drag());
            track.input(ui, &axis_resp, st);
            let painter = ui.painter_at(avail);
            painter.text(egui::pos2(avail.left(), axis.center().y), egui::Align2::LEFT_CENTER, "TOPIC", FontId::proportional(10.5), pal::WEAK);
            let nice = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 21_600.0];
            let want = (view.1 - view.0) / w as f64 * 90.0;
            let step = nice.iter().copied().find(|s| *s >= want).unwrap_or(86_400.0);
            let mut k = (view.0 / step).ceil() as i64;
            while (k as f64) * step <= view.1 {
                let s = k as f64 * step;
                let x = to_x(s);
                painter.line_segment([egui::pos2(x, axis.bottom() - 5.0), egui::pos2(x, axis.bottom())], Stroke::new(1.0f32, pal::WEAK));
                painter.text(egui::pos2(x + 3.0, axis.center().y - 1.0), egui::Align2::LEFT_CENTER, tick_label(s, step), FontId::monospace(10.5), pal::WEAK);
                k += 1;
            }
            painter.line_segment([egui::pos2(track.x0, axis.bottom()), egui::pos2(track.x1, axis.bottom())], Stroke::new(1.0f32, pal::STROKE));

            // One lane per topic.
            ui.spacing_mut().item_spacing.y = 2.0;
            let lanes_top = ui.cursor().top();
            egui::ScrollArea::vertical().auto_shrink(false).show_rows(ui, row_h, rows.len(), |ui, range| {
                for t in &rows[range] {
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), row_h), egui::Sense::click_and_drag());
                    let lane = egui::Rect::from_min_max(egui::pos2(track.x0, rect.top() + 2.0), egui::pos2(track.x1, rect.bottom() - 2.0));
                    let color = t.kind.map_or(pal::WEAK, kind_color);
                    let p = ui.painter();
                    let name_rect = egui::Rect::from_min_max(rect.min, egui::pos2(track.x0 - 8.0, rect.bottom()));
                    p.rect_filled(egui::Rect::from_center_size(egui::pos2(rect.left() + 4.0, rect.center().y), egui::vec2(7.0, 7.0)), 2.0f32, color);
                    p.with_clip_rect(name_rect).text(
                        egui::pos2(rect.left() + 14.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        &t.name,
                        FontId::proportional(12.0),
                        if t.kind.is_some() { pal::TEXT } else { pal::WEAK },
                    );
                    p.rect_filled(lane, 2.0f32, pal::BG);
                    track.density(&p.with_clip_rect(lane), lane, &t.times, view, color);
                    track.input(ui, &resp, st);
                    if resp.hover_pos().is_some_and(|h| h.x < track.x0) {
                        resp.on_hover_text(format!("{}\n{} · {} messages{}", t.name, t.type_name, t.times.len(), if t.kind.is_some() { "" } else { " · not shown by rsviz" }));
                    }
                }
            });

            // Cursor over the axis and every lane, and the time under the pointer.
            let tracks = egui::Rect::from_min_max(egui::pos2(track.x0, axis.top()), egui::pos2(track.x1, avail.bottom()));
            let painter = ui.painter_at(tracks.expand2(egui::vec2(6.0, 0.0)));
            if let Some(h) = ctx.pointer_hover_pos().filter(|h| tracks.contains(*h)) {
                let s = view.0 + ((h.x - track.x0) / w) as f64 * (view.1 - view.0);
                painter.line_segment([egui::pos2(h.x, axis.bottom()), egui::pos2(h.x, tracks.bottom())], Stroke::new(1.0f32, pal::WEAK.gamma_multiply(0.6)));
                let g = painter.layout_no_wrap(clock(s, dur), FontId::monospace(10.5), pal::STRONG);
                let pos = egui::pos2((h.x + 6.0).min(tracks.right() - g.size().x - 2.0), lanes_top + 2.0);
                painter.rect_filled(egui::Rect::from_min_size(pos - egui::vec2(3.0, 1.0), g.size() + egui::vec2(6.0, 2.0)), 3.0f32, pal::ACTIVE);
                painter.galley(pos, g, pal::STRONG);
            }
            let x = to_x(cur_s);
            if x >= track.x0 - 0.5 && x <= track.x1 + 0.5 {
                painter.line_segment([egui::pos2(x, axis.top() + 4.0), egui::pos2(x, tracks.bottom())], Stroke::new(1.5f32, pal::ACCENT));
                let head = vec![egui::pos2(x - 5.0, axis.top() + 2.0), egui::pos2(x + 5.0, axis.top() + 2.0), egui::pos2(x, axis.top() + 10.0)];
                painter.add(egui::Shape::convex_polygon(head, pal::ACCENT, Stroke::NONE));
            }
        });
    close
}
