//! GPU side (wgpu). Point clouds are drawn by "vertex pulling": the raw PointCloud2 point bytes are copied once into a
//! storage buffer and the vertex shader reads x, y, z and the colour at the message's own offsets, so no point is ever
//! touched by the CPU. Each point is a screen-aligned quad of `point_px` pixels (6 vertices, no vertex buffer).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::msgs::CloudColor;
use crate::scene::{MeshVertex, Texture2D};

pub const COLOR_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Gpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    /// `present`: the present modes to use, by order of preference (vsync Fifo when none is supported).
    /// `gpu`: part of the name of the GPU to draw with ("intel", "nvidia"); the most powerful one when None.
    pub async fn new(window: Arc<winit::window::Window>, present: &[wgpu::PresentMode], gpu: Option<&str>) -> Gpu {
        // WGPU_BACKEND=vulkan|gl overrides the backend (default: Vulkan / Metal / DX12).
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..Default::default() });
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let named = gpu.and_then(|want| {
            let all = instance.enumerate_adapters(backends);
            let names: Vec<String> = all.iter().map(|a| a.get_info().name).collect();
            let found = all.into_iter().find(|a| a.get_info().name.to_lowercase().contains(&want.to_lowercase()) && a.is_surface_supported(&surface));
            if found.is_none() {
                log::warn!("no GPU named like \"{want}\" can draw this window (GPUs: {}): using the default one", names.join(", "));
            }
            found
        });
        let adapter = match named {
            Some(a) => a,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("no GPU adapter"),
        };
        log::info!("GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rsviz"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    // Small GPU allocation blocks: the NVIDIA Vulkan driver otherwise maps ~150 MB up front (measured
                    // 280 -> 160 MB resident, same frame rate and CPU). RSVIZ_MEMORY=fast restores the large blocks.
                    memory_hints: if std::env::var("RSVIZ_MEMORY").as_deref() == Ok("fast") { wgpu::MemoryHints::Performance } else { wgpu::MemoryHints::MemoryUsage },
                },
                None,
            )
            .await
            .expect("request device");
        let caps = surface.get_capabilities(&adapter);
        // egui wants a non-sRGB framebuffer.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let present_mode = present.iter().copied().find(|p| caps.present_modes.contains(p)).unwrap_or(wgpu::PresentMode::Fifo);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        log::info!("present mode {present_mode:?}");
        Gpu { surface, device, queue, config }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

/// Orbit camera around `anchor + offset`: the anchor follows a TF frame (or the scene centre), the user's panning is
/// the offset, so a followed drone stays in view while you can still look around it.
pub struct Camera {
    pub anchor: Vec3,
    pub offset: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { anchor: Vec3::ZERO, offset: Vec3::ZERO, yaw: -2.2, pitch: 0.6, dist: 30.0 }
    }
}

impl Camera {
    fn target(&self) -> Vec3 {
        self.anchor + self.offset
    }
    pub fn eye(&self) -> Vec3 {
        self.target() + self.dist * Vec3::new(self.pitch.cos() * self.yaw.cos(), self.pitch.cos() * self.yaw.sin(), self.pitch.sin())
    }
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target(), Vec3::Z);
        let near = (self.dist * 0.002).max(0.01);
        Mat4::perspective_rh(55f32.to_radians(), aspect, near, near * 1e6) * view
    }
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(-1.55, 1.55);
    }
    pub fn pan(&mut self, dx: f32, dy: f32, view_h: f32) {
        let fwd = (self.target() - self.eye()).normalize();
        let right = fwd.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(fwd);
        let k = self.dist * 2.0 * (55f32.to_radians() / 2.0).tan() / view_h.max(1.0);
        self.offset += (-dx * right + dy * up) * k;
    }
    pub fn zoom(&mut self, steps: f32) {
        self.dist = (self.dist * (1.0 - steps * 0.1).clamp(0.5, 1.5)).clamp(0.05, 1e6);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointUniform {
    vp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    viewport: [f32; 2],
    point_px: f32,
    _p0: f32,
    step_w: u32,
    xw: u32,
    yw: u32,
    zw: u32,
    color_mode: u32,
    cw: u32,
    _p1: [u32; 2],
    range: [f32; 2],
    _p2: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

const POINT_WGSL: &str = r#"
struct U {
  vp: mat4x4<f32>, model: mat4x4<f32>,
  viewport: vec2<f32>, point_px: f32, _p0: f32,
  step_w: u32, xw: u32, yw: u32, zw: u32,
  color_mode: u32, cw: u32, _p1: vec2<u32>,
  range: vec2<f32>, _p2: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> data: array<u32>;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32> };

fn lin(c: vec3<f32>) -> vec3<f32> { return pow(c, vec3<f32>(2.2)); }
fn cmap(t: f32) -> vec3<f32> {
  // Compact "turbo"-like ramp, blue -> cyan -> green -> yellow -> red.
  let r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
  let g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
  let b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
  return vec3<f32>(r, g, b);
}

// The instance index carries the stride of the draw: only one point out of `stride` is drawn (a cloud seen from far).
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) stride: u32) -> VOut {
  var o: VOut;
  let i = (vi / 6u) * max(stride, 1u);
  let c = vi % 6u;
  let b = i * u.step_w;
  let xb = data[b + u.xw];
  if ((xb & 0x7f800000u) == 0x7f800000u) { o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0); o.col = vec3<f32>(0.0); return o; }
  let p = vec3<f32>(bitcast<f32>(xb), bitcast<f32>(data[b + u.yw]), bitcast<f32>(data[b + u.zw]));
  let w = u.model * vec4<f32>(p, 1.0);
  var clip = u.vp * w;
  var k = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  clip = vec4<f32>(clip.xy + k[c] * u.point_px / u.viewport * clip.w, clip.zw);
  var col: vec3<f32>;
  if (u.color_mode == 1u) {
    let v = data[b + u.cw];
    col = lin(vec3<f32>(f32((v >> 16u) & 255u), f32((v >> 8u) & 255u), f32(v & 255u)) / 255.0);
  } else {
    var s = p.z;
    if (u.color_mode == 2u) { s = bitcast<f32>(data[b + u.cw]); }
    col = lin(cmap(clamp((s - u.range.x) / max(u.range.y - u.range.x, 1e-6), 0.0, 1.0)));
  }
  o.pos = clip;
  o.col = col;
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> { return vec4<f32>(i.col, 1.0); }
"#;

const LINE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> vp: mat4x4<f32>;
struct VIn { @location(0) pos: vec3<f32>, @location(1) col: vec4<f32> };
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec4<f32> };
@vertex fn vs(v: VIn) -> VOut { var o: VOut; o.pos = vp * vec4<f32>(v.pos, 1.0); o.col = v.col; return o; }
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> { return vec4<f32>(pow(i.col.rgb, vec3<f32>(2.2)), i.col.a); }
"#;

const MESH_WGSL: &str = r#"
struct MU { vp: mat4x4<f32>, model: mat4x4<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> mu: MU;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
struct VIn { @location(0) pos: vec3<f32>, @location(1) col: vec4<f32>, @location(2) uv: vec2<f32> };
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec4<f32>, @location(1) uv: vec2<f32>, @location(2) wp: vec3<f32> };
@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  let w = mu.model * vec4<f32>(v.pos, 1.0);
  o.pos = mu.vp * w; o.col = v.col; o.uv = v.uv; o.wp = w.xyz;
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
  var c = vec4<f32>(pow(clamp(i.col.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2)), i.col.a);
  let t = textureSample(tex, smp, i.uv);
  if (mu.flags.y > 0.5) { c = c * t; }
  // Flat shading from the screen-space derivatives (no normals stored); textured meshes keep their true colours.
  let n = normalize(cross(dpdx(i.wp), dpdy(i.wp)));
  if (mu.flags.x > 0.5) { c = vec4<f32>(c.rgb * (0.35 + 0.65 * abs(dot(n, normalize(vec3<f32>(0.35, 0.25, 0.9))))), c.a); }
  return c;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshUniform {
    vp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    flags: [f32; 4],
}

struct GpuMesh {
    vbuf: wgpu::Buffer,
    count: u32,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    version: u64,
    shade: bool,
    textured: bool,
}

/// A mesh to draw this frame (marker or robot visual); vertices and texture are shared, not copied.
pub struct MeshSnapshot {
    pub verts: Arc<Vec<MeshVertex>>,
    pub texture: Option<Arc<Texture2D>>,
    pub shade: bool,
    pub model: Mat4,
    pub version: u64,
}

struct GpuCloud {
    data: wgpu::Buffer,
    cap: u64,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    n: u32,
    version: u64,
    /// What is drawn this frame: (first point slot, end slot, stride); point = slot * stride.
    draws: Vec<(u32, u32, u32)>,
}

/// Everything the 3D view needs from a cloud, cloned under the scene lock (the bytes are an Arc, not copied).
pub struct CloudSnapshot {
    pub bytes: Arc<Vec<u8>>,
    pub off: usize,
    pub len: usize,
    pub n: u32,
    pub point_step: u32,
    pub xyz: [u32; 3],
    pub color: CloudColor,
    pub range: (f32, f32),
    pub chunks: Arc<Vec<crate::msgs::Chunk>>,
    pub version: u64,
    pub model: Mat4,
}

pub struct View3D {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    pub size: (u32, u32),
    pub egui_id: egui::TextureId,
    point_pipe: wgpu::RenderPipeline,
    point_bgl: wgpu::BindGroupLayout,
    line_pipe: wgpu::RenderPipeline,
    line_uniform: wgpu::Buffer,
    line_bind: wgpu::BindGroup,
    line_buf: wgpu::Buffer,
    line_cap: u64,
    line_count: u32,
    clouds: HashMap<String, GpuCloud>,
    /// Cloud points drawn in the last frame, and points of the clouds shown.
    pub cloud_points: (u64, u64),
    mesh_pipe: wgpu::RenderPipeline,
    mesh_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    white: wgpu::TextureView,
    meshes: HashMap<String, GpuMesh>,
}

fn targets(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let mk = |fmt, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        })
    };
    let color = mk(COLOR_FMT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
    let depth = mk(DEPTH_FMT, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let cv = color.create_view(&Default::default());
    let dv = depth.create_view(&Default::default());
    (color, cv, dv)
}

impl View3D {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, egui_rend: &mut egui_wgpu::Renderer) -> View3D {
        let (color, color_view, depth_view) = targets(device, 64, 64);
        let egui_id = egui_rend.register_native_texture(device, &color_view, wgpu::FilterMode::Linear);
        let depth_state = Some(wgpu::DepthStencilState {
            format: DEPTH_FMT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        });
        let target = [Some(wgpu::ColorTargetState { format: COLOR_FMT, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })];

        let point_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("points"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let pm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("points"), source: wgpu::ShaderSource::Wgsl(POINT_WGSL.into()) });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&point_bgl], push_constant_ranges: &[] });
        let point_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &pm, entry_point: "vs", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &pm, entry_point: "fs", compilation_options: Default::default(), targets: &target }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: depth_state.clone(),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let lbgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lines"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let lm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("lines"), source: wgpu::ShaderSource::Wgsl(LINE_WGSL.into()) });
        let ll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&lbgl], push_constant_ranges: &[] });
        let line_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&ll),
            vertex: wgpu::VertexState {
                module: &lm,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState { module: &lm, entry_point: "fs", compilation_options: Default::default(), targets: &target }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: depth_state,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let line_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lines vp"),
            contents: bytemuck::bytes_of(&Mat4::IDENTITY.to_cols_array_2d()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let line_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &lbgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: line_uniform.as_entire_binding() }],
        });
        let line_cap = 1 << 16;
        let line_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lines"),
            size: line_cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let mm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mesh"), source: wgpu::ShaderSource::Wgsl(MESH_WGSL.into()) });
        let ml = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&mesh_bgl], push_constant_ranges: &[] });
        let mesh_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: Some(&ml),
            vertex: wgpu::VertexState {
                module: &mm,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState { module: &mm, entry_point: "fs", compilation_options: Default::default(), targets: &target }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FMT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let white = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("white"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255u8; 4],
            )
            .create_view(&Default::default());
        View3D {
            mesh_pipe,
            mesh_bgl,
            sampler,
            white,
            meshes: HashMap::new(),
            color,
            color_view,
            depth_view,
            size: (64, 64),
            egui_id,
            point_pipe,
            point_bgl,
            line_pipe,
            line_uniform,
            line_bind,
            line_buf,
            line_cap,
            line_count: 0,
            clouds: HashMap::new(),
            cloud_points: (0, 0),
        }
    }

    pub fn ensure_size(&mut self, device: &wgpu::Device, egui_rend: &mut egui_wgpu::Renderer, w: u32, h: u32) {
        let (w, h) = (w.max(16), h.max(16));
        if (w, h) != self.size {
            let (c, cv, dv) = targets(device, w, h);
            self.color = c;
            self.color_view = cv;
            self.depth_view = dv;
            self.size = (w, h);
            egui_rend.update_egui_texture_from_wgpu_texture(device, &self.color_view, wgpu::FilterMode::Linear, self.egui_id);
        }
    }

    pub fn set_lines(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[LineVertex]) {
        let bytes: &[u8] = bytemuck::cast_slice(verts);
        if bytes.len() as u64 > self.line_cap {
            self.line_cap = (bytes.len() as u64).next_power_of_two();
            self.line_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lines"),
                size: self.line_cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.line_buf, 0, bytes);
        }
        self.line_count = verts.len() as u32;
    }

    /// Uploads new point data (one copy of the message bytes into the GPU buffer) and refreshes every cloud uniform.
    pub fn sync_clouds(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, snaps: &HashMap<String, CloudSnapshot>, vp: Mat4, eye: Vec3, point_px: f32) {
        self.clouds.retain(|k, _| snaps.contains_key(k));
        self.cloud_points = (0, 0);
        // RSVIZ_NO_LOD=1 draws every point of every cloud (to compare).
        let lod = std::env::var_os("RSVIZ_NO_LOD").is_none();
        // A cloud bigger than what this GPU binds as one buffer is drawn in part (its first points), not at all refused.
        let limits = device.limits();
        let max = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) & !3;
        for (topic, s) in snaps {
            let step = s.point_step.max(1) as u64;
            let len = (s.len as u64).min(max / step * step) as usize;
            let n = s.n.min((len as u64 / step) as u32);
            let need = ((len as u64 + 3) & !3).max(16);
            let fresh = !self.clouds.contains_key(topic) || self.clouds[topic].cap < need;
            if fresh {
                if len < s.len {
                    log::warn!("{topic}: cloud of {} MB, this GPU takes {} MB per buffer: {n} of {} points drawn", s.len >> 20, max >> 20, s.n);
                }
                let cap = ((need + need / 4 + 3) & !3).min(max).max(need);
                let data = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cloud"),
                    size: cap,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cloud u"),
                    size: std::mem::size_of::<PointUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.point_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: data.as_entire_binding() },
                    ],
                });
                self.clouds.insert(topic.clone(), GpuCloud { data, cap, uniform, bind, n: 0, version: 0, draws: Vec::new() });
            }
            let g = self.clouds.get_mut(topic).unwrap();
            if fresh || g.version != s.version {
                let src = &s.bytes[s.off..s.off + len];
                if src.len() % 4 == 0 {
                    queue.write_buffer(&g.data, 0, src);
                } else {
                    let mut pad = src.to_vec();
                    pad.resize((src.len() + 3) & !3, 0);
                    queue.write_buffer(&g.data, 0, &pad);
                }
                g.version = s.version;
                g.n = n;
            }
            // Big clouds: the chunks out of the view are skipped, and of a chunk seen from far only one point out of
            // `stride` is drawn, as long as the points drawn still touch each other on screen (same picture, less work).
            g.draws.clear();
            if s.chunks.is_empty() || !lod {
                g.draws.push((0, g.n, 1));
            } else {
                let mvp = vp * s.model;
                let eye_local = s.model.inverse().transform_point3(eye);
                let focal_px = self.size.1 as f32 * 0.5 / (55f32.to_radians() * 0.5).tan();
                for c in s.chunks.iter() {
                    if c.first >= g.n {
                        break;
                    }
                    let (lo, hi) = (Vec3::from(c.min), Vec3::from(c.max));
                    // Outside one plane of the view volume with its 8 corners = not visible.
                    let mut out = [true; 6];
                    for k in 0..8 {
                        let p = mvp * Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z }).extend(1.0);
                        let m = 1.05 * p.w.abs();
                        out[0] &= p.x < -m;
                        out[1] &= p.x > m;
                        out[2] &= p.y < -m;
                        out[3] &= p.y > m;
                        out[4] &= p.z < 0.0;
                        out[5] &= p.z > p.w;
                    }
                    if out.iter().any(|o| *o) {
                        continue;
                    }
                    // Mean spacing of the points (they cover about the two largest sides of the box), in pixels at
                    // the nearest point of the box.
                    let mut e = (hi - lo).to_array();
                    e.sort_by(|a, b| b.total_cmp(a));
                    let spacing = if e[1] > 1e-6 { (e[0] * e[1] / c.count as f32).sqrt() } else { e[0] / c.count as f32 };
                    let dist = (eye_local - eye_local.clamp(lo, hi)).length().max(1e-3);
                    let px = spacing * focal_px / dist;
                    let stride = if px > 0.0 { ((point_px / px).floor() as u32).clamp(1, 64) } else { 1 };
                    let end = (c.first + c.count).min(g.n);
                    g.draws.push((c.first.div_ceil(stride), end.div_ceil(stride), stride));
                }
            }
            self.cloud_points.0 += g.draws.iter().map(|d| (d.1 - d.0) as u64).sum::<u64>();
            self.cloud_points.1 += g.n as u64;
            let (color_mode, cw) = match s.color {
                CloudColor::Height => (0, 0),
                CloudColor::Rgb(o) => (1, o / 4),
                CloudColor::Scalar(o) => (2, o / 4),
            };
            let u = PointUniform {
                vp: vp.to_cols_array_2d(),
                model: s.model.to_cols_array_2d(),
                viewport: [self.size.0 as f32, self.size.1 as f32],
                point_px,
                _p0: 0.0,
                step_w: s.point_step / 4,
                xw: s.xyz[0] / 4,
                yw: s.xyz[1] / 4,
                zw: s.xyz[2] / 4,
                color_mode,
                cw,
                _p1: [0; 2],
                range: [s.range.0, s.range.1],
                _p2: [0.0; 2],
            };
            queue.write_buffer(&g.uniform, 0, bytemuck::bytes_of(&u));
        }
    }

    /// (Re)creates GPU buffers of new or changed meshes (vertices and texture are uploaded once) and updates every uniform.
    pub fn sync_meshes(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, snaps: &HashMap<String, MeshSnapshot>, vp: Mat4) {
        self.meshes.retain(|k, _| snaps.contains_key(k));
        for (key, s) in snaps {
            if s.verts.is_empty() {
                self.meshes.remove(key);
                continue;
            }
            if self.meshes.get(key).map_or(true, |g| g.version != s.version) {
                let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh"),
                    contents: bytemuck::cast_slice(s.verts.as_slice()),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let tex_view = s.texture.as_ref().map(|t| {
                    device
                        .create_texture_with_data(
                            queue,
                            &wgpu::TextureDescriptor {
                                label: Some("marker texture"),
                                size: wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                                view_formats: &[],
                            },
                            wgpu::util::TextureDataOrder::LayerMajor,
                            &t.rgba,
                        )
                        .create_view(&Default::default())
                });
                let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mesh u"),
                    size: std::mem::size_of::<MeshUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.mesh_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(tex_view.as_ref().unwrap_or(&self.white)) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                });
                self.meshes.insert(
                    key.clone(),
                    GpuMesh { vbuf, count: s.verts.len() as u32, uniform, bind, version: s.version, shade: s.shade, textured: s.texture.is_some() },
                );
            }
            let g = &self.meshes[key];
            let u = MeshUniform {
                vp: vp.to_cols_array_2d(),
                model: s.model.to_cols_array_2d(),
                flags: [if g.shade { 1.0 } else { 0.0 }, if g.textured { 1.0 } else { 0.0 }, 0.0, 0.0],
            };
            queue.write_buffer(&g.uniform, 0, bytemuck::bytes_of(&u));
        }
    }

    pub fn render(&self, enc: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, vp: Mat4, bg: [f64; 3]) {
        queue.write_buffer(&self.line_uniform, 0, bytemuck::bytes_of(&vp.to_cols_array_2d()));
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3d"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0], g: bg[1], b: bg[2], a: 1.0 }), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if !self.meshes.is_empty() {
            rp.set_pipeline(&self.mesh_pipe);
            for g in self.meshes.values() {
                rp.set_bind_group(0, &g.bind, &[]);
                rp.set_vertex_buffer(0, g.vbuf.slice(..));
                rp.draw(0..g.count, 0..1);
            }
        }
        if self.line_count > 0 {
            rp.set_pipeline(&self.line_pipe);
            rp.set_bind_group(0, &self.line_bind, &[]);
            rp.set_vertex_buffer(0, self.line_buf.slice(..));
            rp.draw(0..self.line_count, 0..1);
        }
        rp.set_pipeline(&self.point_pipe);
        for g in self.clouds.values() {
            if g.n > 0 && !g.draws.is_empty() {
                rp.set_bind_group(0, &g.bind, &[]);
                for (a, b, stride) in &g.draws {
                    if b > a {
                        rp.draw(a * 6..b * 6, *stride..*stride + 1);
                    }
                }
            }
        }
    }
}

/// An image topic's texture, registered with egui so the panel shows it without any extra copy.
pub struct ImageTex {
    tex: wgpu::Texture,
    pub id: egui::TextureId,
    pub size: (u32, u32),
    pub version: u64,
}

impl ImageTex {
    pub fn update(slot: &mut Option<ImageTex>, device: &wgpu::Device, queue: &wgpu::Queue, egui_rend: &mut egui_wgpu::Renderer, w: u32, h: u32, rgba: &[u8], version: u64) {
        if slot.as_ref().map_or(true, |t| t.size != (w, h)) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("image"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let id = match slot.take() {
                Some(old) => {
                    egui_rend.update_egui_texture_from_wgpu_texture(device, &view, wgpu::FilterMode::Linear, old.id);
                    old.id
                }
                None => egui_rend.register_native_texture(device, &view, wgpu::FilterMode::Linear),
            };
            *slot = Some(ImageTex { tex, id, size: (w, h), version: 0 });
        }
        let t = slot.as_mut().unwrap();
        if t.version != version {
            queue.write_texture(
                wgpu::ImageCopyTexture { texture: &t.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                rgba,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            t.version = version;
        }
    }
}
