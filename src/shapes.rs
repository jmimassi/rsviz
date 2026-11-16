//! Triangle geometry for markers and the robot model, built once per message on the ROS thread (never per frame).

use std::sync::Arc;

use glam::{DAffine3, DQuat, DVec3, Vec3};

use crate::msgs::Marker;
use crate::scene::{MeshVertex, Prim, Texture2D};

fn v(p: Vec3, c: [f32; 4]) -> MeshVertex {
    MeshVertex { pos: p.to_array(), color: c, uv: [0.0, 0.0] }
}

/// Axis-aligned box of size s centred on c.
pub fn cube(out: &mut Vec<MeshVertex>, c: Vec3, s: Vec3, col: [f32; 4]) {
    let h = s * 0.5;
    let p = |x: f32, y: f32, z: f32| c + Vec3::new(x * h.x, y * h.y, z * h.z);
    let faces = [
        [p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.)],
        [p(-1., -1., -1.), p(-1., 1., -1.), p(1., 1., -1.), p(1., -1., -1.)],
        [p(-1., -1., -1.), p(1., -1., -1.), p(1., -1., 1.), p(-1., -1., 1.)],
        [p(-1., 1., -1.), p(-1., 1., 1.), p(1., 1., 1.), p(1., 1., -1.)],
        [p(-1., -1., -1.), p(-1., -1., 1.), p(-1., 1., 1.), p(-1., 1., -1.)],
        [p(1., -1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(1., -1., 1.)],
    ];
    for f in faces {
        for i in [0, 1, 2, 0, 2, 3] {
            out.push(v(f[i], col));
        }
    }
}

/// Ellipsoid of diameters s centred on c.
pub fn sphere(out: &mut Vec<MeshVertex>, c: Vec3, s: Vec3, col: [f32; 4]) {
    let (nu, nv) = (16, 10);
    let pt = |i: usize, j: usize| {
        let th = i as f32 / nu as f32 * std::f32::consts::TAU;
        let ph = j as f32 / nv as f32 * std::f32::consts::PI;
        c + 0.5 * s * Vec3::new(ph.sin() * th.cos(), ph.sin() * th.sin(), ph.cos())
    };
    for i in 0..nu {
        for j in 0..nv {
            let (a, b, cc, d) = (pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1));
            for p in [a, b, cc, a, cc, d] {
                out.push(v(p, col));
            }
        }
    }
}

/// Cylinder (r0 at a, r1 at b; r1 = 0 gives a cone) with caps.
pub fn frustum(out: &mut Vec<MeshVertex>, a: Vec3, b: Vec3, r0: f32, r1: f32, col: [f32; 4]) {
    let axis = b - a;
    if axis.length_squared() < 1e-12 {
        return;
    }
    let d = axis.normalize();
    let u = d.any_orthonormal_vector();
    let w = d.cross(u);
    let n = 16;
    let ring = |r: f32, o: Vec3, k: usize| {
        let t = k as f32 / n as f32 * std::f32::consts::TAU;
        o + r * (t.cos() * u + t.sin() * w)
    };
    for k in 0..n {
        let (p0, p1, q0, q1) = (ring(r0, a, k), ring(r0, a, k + 1), ring(r1, b, k), ring(r1, b, k + 1));
        for p in [p0, p1, q1, p0, q1, q0] {
            out.push(v(p, col));
        }
        for p in [a, p1, p0] {
            out.push(v(p, col));
        }
        if r1 > 0.0 {
            for p in [b, q0, q1] {
                out.push(v(p, col));
            }
        }
    }
}

pub fn arrow(out: &mut Vec<MeshVertex>, a: Vec3, b: Vec3, shaft_d: f32, head_d: f32, head_l: f32, col: [f32; 4]) {
    let len = (b - a).length();
    if len < 1e-9 {
        return;
    }
    let d = (b - a) / len;
    let hl = head_l.min(len);
    let m = b - d * hl;
    frustum(out, a, m, shaft_d * 0.5, shaft_d * 0.5, col);
    frustum(out, m, b, head_d * 0.5, 0.0, col);
}

fn clamp_color(c: [f32; 4]) -> [f32; 4] {
    [c[0].clamp(0., 1.), c[1].clamp(0., 1.), c[2].clamp(0., 1.), c[3].clamp(0., 1.)]
}

/// Decoded pixels (1, 3 or 4 channels, as the JPEG decoder returns them) -> RGBA8.
pub fn to_rgba(px: Vec<u8>, w: usize, h: usize) -> Option<Vec<u8>> {
    let n = w * h;
    if n == 0 || px.len() % n != 0 {
        return None;
    }
    Some(match px.len() / n {
        4 => px,
        3 => px.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        1 => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        2 => px.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        _ => return None,
    })
}

/// Decodes an embedded marker texture (JPEG; PNG is not supported).
fn decode_texture(bytes: &[u8]) -> Option<Arc<Texture2D>> {
    use zune_core::{colorspace::ColorSpace, options::DecoderOptions};
    if bytes.is_empty() {
        return None;
    }
    let mut d = zune_jpeg::JpegDecoder::new_with_options(bytes, DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA).set_max_width(16384).set_max_height(16384));
    let px = d.decode().ok()?;
    let info = d.info()?;
    let rgba = to_rgba(px, info.width as usize, info.height as usize)?;
    Some(Arc::new(Texture2D { width: info.width as u32, height: info.height as u32, rgba }))
}

/// marker frame <- marker pose
pub fn marker_pose(m: &Marker) -> DAffine3 {
    DAffine3::from_rotation_translation(DQuat::from_xyzw(m.quat[0], m.quat[1], m.quat[2], m.quat[3]).normalize(), DVec3::from(m.pos))
}

/// The drawable of a Marker (ADD/MODIFY), or an error message for the types rsviz does not draw.
pub fn marker_prim(m: &Marker) -> Result<Prim, String> {
    let col = clamp_color(m.color);
    let sc = Vec3::new(m.scale[0] as f32, m.scale[1] as f32, m.scale[2] as f32);
    let pcol = |i: usize| m.colors.get(i).map(|c| clamp_color(*c)).unwrap_or(col);
    let pts: Vec<Vec3> = m.points.iter().map(|p| Vec3::from(*p)).collect();
    let mut verts = Vec::new();
    match m.kind {
        0 => {
            if pts.len() >= 2 {
                let hl = if sc.z > 0.0 { sc.z } else { 0.23 * (pts[1] - pts[0]).length() };
                arrow(&mut verts, pts[0], pts[1], sc.x, sc.y.max(sc.x), hl, col);
            } else {
                arrow(&mut verts, Vec3::ZERO, Vec3::X * sc.x, sc.y, sc.z.max(sc.y) * 1.6, 0.23 * sc.x, col);
            }
        }
        1 => cube(&mut verts, Vec3::ZERO, sc, col),
        2 => sphere(&mut verts, Vec3::ZERO, sc, col),
        3 => frustum(&mut verts, Vec3::new(0., 0., -sc.z / 2.), Vec3::new(0., 0., sc.z / 2.), sc.x / 2., sc.x / 2., col),
        4 | 5 => {
            let (mut p, mut c) = (Vec::new(), Vec::new());
            let pairs: Vec<(usize, usize)> = if m.kind == 4 { (1..pts.len()).map(|i| (i - 1, i)).collect() } else { (0..pts.len() / 2).map(|i| (2 * i, 2 * i + 1)).collect() };
            for (a, b) in pairs {
                p.push(m.points[a]);
                p.push(m.points[b]);
                c.push(pcol(a));
                c.push(pcol(b));
            }
            return Ok(Prim::Lines(p, c));
        }
        6 | 7 | 8 => {
            let s = if m.kind == 8 { Vec3::new(sc.x, sc.y.max(sc.x), sc.x.min(sc.y.max(1e-3))) } else { sc };
            for (i, p) in pts.iter().enumerate() {
                if m.kind == 7 { sphere(&mut verts, *p, s, pcol(i)) } else { cube(&mut verts, *p, s, pcol(i)) }
            }
        }
        9 => return Ok(Prim::Text { pos: [0.0; 3], text: m.text.clone(), color: col }),
        11 => {
            let texture = decode_texture(&m.texture);
            for (i, p) in m.points.iter().enumerate() {
                verts.push(MeshVertex { pos: *p, color: pcol(i), uv: m.uv.get(i).copied().unwrap_or([0.0, 0.0]) });
            }
            verts.truncate(verts.len() / 3 * 3);
            let shade = texture.is_none();
            return Ok(Prim::Mesh { verts: Arc::new(verts), texture, shade });
        }
        10 => return Err(format!("MESH_RESOURCE markers are not drawn ({})", m.mesh_resource)),
        k => return Err(format!("marker type {k} is not drawn")),
    }
    Ok(Prim::Mesh { verts: Arc::new(verts), texture: None, shade: true })
}
