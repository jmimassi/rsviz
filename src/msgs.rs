//! Decoders for the supported message types, straight from the serialized CDR bytes. Bulk payloads (point data, image
//! pixels) are never copied here: they are returned as (offset, length) into the received buffer.

use crate::cdr::{Cdr, CdrError, Result};

pub struct Header {
    #[allow(dead_code)]
    pub stamp: f64,
    pub frame_id: String,
}

fn header(c: &mut Cdr) -> Result<Header> {
    if c.ros1() {
        c.u32()?; // seq
    }
    let sec = c.i32()?;
    let nsec = c.u32()?;
    let frame_id = c.string()?.to_owned();
    Ok(Header { stamp: sec as f64 + nsec as f64 * 1e-9, frame_id })
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CloudColor {
    /// Height (z) mapped through a colormap.
    Height,
    /// Packed 0x00RRGGBB at this byte offset (float32 or uint32 "rgb"/"rgba" field).
    Rgb(u32),
    /// float32 scalar (e.g. intensity) at this byte offset, mapped through a colormap.
    Scalar(u32),
}

/// A PointCloud2 whose point data stays in the received buffer, read by the GPU as is when `gpu_direct`.
pub struct Cloud {
    pub header: Header,
    pub n: u32,
    pub point_step: u32,
    pub xyz: [u32; 3],
    pub color: CloudColor,
    pub data_off: usize,
    pub data_len: usize,
    /// Range used by the colormap (height or scalar), from a sample of points.
    pub range: (f32, f32),
}

/// A run of consecutive points of a big cloud and the box that holds them (in the cloud's frame). The renderer skips
/// the runs outside the view and draws only part of the points of the runs seen from far.
#[derive(Clone, Copy)]
pub struct Chunk {
    pub first: u32,
    pub count: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Points per chunk, and the size from which a cloud is cut in chunks (smaller clouds are always drawn whole).
pub const CHUNK_POINTS: u32 = 65_536;
pub const CHUNKED_FROM: u32 = 262_144;

/// Chunks of a cloud whose x y z are little-endian float32 at byte offsets `xyz` of each `step`-byte point (the layout
/// kept for the GPU). One pass over the points; empty for a cloud under `CHUNKED_FROM` points.
pub fn cloud_chunks(data: &[u8], n: u32, step: u32, xyz: [u32; 3]) -> Vec<Chunk> {
    if n < CHUNKED_FROM || step == 0 {
        return Vec::new();
    }
    let (step, o) = (step as usize, [xyz[0] as usize, xyz[1] as usize, xyz[2] as usize]);
    let mut out = Vec::with_capacity((n / CHUNK_POINTS + 1) as usize);
    let mut first = 0u32;
    while first < n {
        let count = CHUNK_POINTS.min(n - first);
        let (mut min, mut max) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in data[first as usize * step..(first + count) as usize * step].chunks_exact(step) {
            let v = [0, 1, 2].map(|a| f32::from_le_bytes(p[o[a]..o[a] + 4].try_into().unwrap()));
            if v[0].is_finite() && v[1].is_finite() && v[2].is_finite() {
                for a in 0..3 {
                    min[a] = min[a].min(v[a]);
                    max[a] = max[a].max(v[a]);
                }
            }
        }
        if min[0] <= max[0] {
            out.push(Chunk { first, count, min, max });
        }
        first += count;
    }
    out
}

const FLOAT32: u8 = 7;
const FLOAT64: u8 = 8;
const UINT32: u8 = 6;

fn read_num(b: &[u8], off: usize, dt: u8, le: bool) -> f32 {
    let get4 = |o: usize| -> [u8; 4] { b[o..o + 4].try_into().unwrap() };
    match dt {
        FLOAT32 => if le { f32::from_le_bytes(get4(off)) } else { f32::from_be_bytes(get4(off)) },
        FLOAT64 => {
            let a: [u8; 8] = b[off..off + 8].try_into().unwrap();
            (if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }) as f32
        }
        1 => b[off] as i8 as f32,
        2 => b[off] as f32,
        3 => { let a = [b[off], b[off + 1]]; (if le { i16::from_le_bytes(a) } else { i16::from_be_bytes(a) }) as f32 }
        4 => { let a = [b[off], b[off + 1]]; (if le { u16::from_le_bytes(a) } else { u16::from_be_bytes(a) }) as f32 }
        5 => (if le { i32::from_le_bytes(get4(off)) } else { i32::from_be_bytes(get4(off)) }) as f32,
        6 => (if le { u32::from_le_bytes(get4(off)) } else { u32::from_be_bytes(get4(off)) }) as f32,
        _ => f32::NAN,
    }
}

/// Decoded cloud plus, when the layout is not GPU-readable as is (float64 coordinates, unaligned fields, big endian),
/// a repacked buffer [x y z color] (16 bytes per point) that replaces the original point data.
pub fn point_cloud2(buf: &[u8]) -> Result<(Cloud, Option<Vec<u8>>)> {
    let mut c = Cdr::new(buf)?;
    let header = header(&mut c)?;
    let height = c.u32()?;
    let width = c.u32()?;
    let nf = c.count(13)?;
    let mut fields = Vec::with_capacity(nf);
    for _ in 0..nf {
        let name = c.string()?.to_owned();
        let offset = c.u32()?;
        let dt = c.u8()?;
        let _count = c.u32()?;
        fields.push((name, offset, dt));
    }
    let big = c.u8()? != 0;
    let point_step = c.u32()?;
    let _row_step = c.u32()?;
    let (data_off, data_len) = c.byte_seq()?;
    let le = !big;
    let n = (height as u64 * width as u64).min(if point_step > 0 { (data_len / point_step as usize) as u64 } else { 0 }) as u32;
    let find = |names: &[&str]| fields.iter().find(|f| names.contains(&f.0.as_str())).cloned();
    let (Some(fx), Some(fy), Some(fz)) = (find(&["x"]), find(&["y"]), find(&["z"])) else {
        return Err(CdrError("PointCloud2 without x/y/z fields"));
    };
    let rgb = find(&["rgb", "rgba"]).filter(|f| f.2 == FLOAT32 || f.2 == UINT32);
    let scalar = find(&["intensity", "i", "reflectivity"]);
    let data = &buf[data_off..data_off + data_len];
    let ps = point_step as usize;

    let direct = le
        && cfg!(target_endian = "little")
        && [&fx, &fy, &fz].iter().all(|f| f.2 == FLOAT32 && f.1 % 4 == 0)
        && point_step % 4 == 0
        && rgb.as_ref().map_or(true, |f| f.1 % 4 == 0)
        && scalar.as_ref().map_or(true, |f| f.2 == FLOAT32 && f.1 % 4 == 0);

    // Colormap range from at most 4096 points (no full pass over the cloud).
    let stride = (n as usize / 4096).max(1);
    let (key_off, key_dt) = match (&rgb, &scalar) {
        (Some(_), _) => (fz.1, fz.2),
        (None, Some(s)) => (s.1, s.2),
        _ => (fz.1, fz.2),
    };
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in (0..n as usize).step_by(stride) {
        let v = read_num(data, i * ps + key_off as usize, key_dt, le);
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() {
        (lo, hi) = (0.0, 1.0);
    }

    if direct {
        let color = match (&rgb, &scalar) {
            (Some(f), _) => CloudColor::Rgb(f.1),
            (None, Some(f)) => CloudColor::Scalar(f.1),
            _ => CloudColor::Height,
        };
        let cloud = Cloud { header, n, point_step, xyz: [fx.1, fy.1, fz.1], color, data_off, data_len, range: (lo, hi) };
        return Ok((cloud, None));
    }

    // Repack once on the CPU: [x y z c] float32, c = packed rgb bits or scalar.
    let mut out = vec![0u8; n as usize * 16];
    for i in 0..n as usize {
        let b = i * ps;
        let p = [read_num(data, b + fx.1 as usize, fx.2, le), read_num(data, b + fy.1 as usize, fy.2, le), read_num(data, b + fz.1 as usize, fz.2, le)];
        let cbits = match (&rgb, &scalar) {
            (Some(f), _) => { let a: [u8; 4] = data[b + f.1 as usize..b + f.1 as usize + 4].try_into().unwrap(); if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) } }
            (None, Some(f)) => read_num(data, b + f.1 as usize, f.2, le).to_bits(),
            _ => 0,
        };
        let o = &mut out[i * 16..i * 16 + 16];
        o[0..4].copy_from_slice(&p[0].to_le_bytes());
        o[4..8].copy_from_slice(&p[1].to_le_bytes());
        o[8..12].copy_from_slice(&p[2].to_le_bytes());
        o[12..16].copy_from_slice(&cbits.to_le_bytes());
    }
    let color = match (&rgb, &scalar) {
        (Some(_), _) => CloudColor::Rgb(12),
        (None, Some(_)) => CloudColor::Scalar(12),
        _ => CloudColor::Height,
    };
    let len = out.len();
    let cloud = Cloud { header, n, point_step: 16, xyz: [0, 4, 8], color, data_off: 0, data_len: len, range: (lo, hi) };
    Ok((cloud, Some(out)))
}

pub struct RawImage {
    #[allow(dead_code)]
    pub header: Header,
    pub width: u32,
    pub height: u32,
    pub encoding: String,
    pub big_endian: bool,
    pub step: u32,
    pub data_off: usize,
    pub data_len: usize,
}

pub fn image(buf: &[u8]) -> Result<RawImage> {
    let mut c = Cdr::new(buf)?;
    let header = header(&mut c)?;
    let height = c.u32()?;
    let width = c.u32()?;
    let encoding = c.string()?.to_owned();
    let big_endian = c.u8()? != 0;
    let step = c.u32()?;
    let (data_off, data_len) = c.byte_seq()?;
    Ok(RawImage { header, width, height, encoding, big_endian, step, data_off, data_len })
}

pub struct CompressedImage {
    #[allow(dead_code)]
    pub header: Header,
    pub format: String,
    pub data_off: usize,
    pub data_len: usize,
}

pub fn compressed_image(buf: &[u8]) -> Result<CompressedImage> {
    let mut c = Cdr::new(buf)?;
    let header = header(&mut c)?;
    let format = c.string()?.to_owned();
    let (data_off, data_len) = c.byte_seq()?;
    Ok(CompressedImage { header, format, data_off, data_len })
}

fn pose(c: &mut Cdr) -> Result<([f64; 3], [f64; 4])> {
    let p = [c.f64()?, c.f64()?, c.f64()?];
    let q = [c.f64()?, c.f64()?, c.f64()?, c.f64()?];
    Ok((p, q))
}

/// nav_msgs/Path: frame and positions.
pub fn path(buf: &[u8]) -> Result<(Header, Vec<[f64; 3]>)> {
    let mut c = Cdr::new(buf)?;
    let h = header(&mut c)?;
    let n = c.count(68)?;
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        header(&mut c)?;
        pts.push(pose(&mut c)?.0);
    }
    Ok((h, pts))
}

/// geometry_msgs/PoseStamped: frame and pose.
pub fn pose_stamped(buf: &[u8]) -> Result<(Header, [f64; 3], [f64; 4])> {
    let mut c = Cdr::new(buf)?;
    let h = header(&mut c)?;
    let (p, q) = pose(&mut c)?;
    Ok((h, p, q))
}

/// nav_msgs/Odometry: frame and pose (twist ignored).
pub fn odometry(buf: &[u8]) -> Result<(Header, [f64; 3], [f64; 4])> {
    let mut c = Cdr::new(buf)?;
    let h = header(&mut c)?;
    let _child = c.string()?;
    let (p, q) = pose(&mut c)?;
    Ok((h, p, q))
}

pub struct Transform {
    pub parent: String,
    pub child: String,
    pub t: [f64; 3],
    /// x y z w
    pub q: [f64; 4],
}

pub fn tf_message(buf: &[u8]) -> Result<Vec<Transform>> {
    let mut c = Cdr::new(buf)?;
    let n = c.count(72)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let h = header(&mut c)?;
        let child = c.string()?.to_owned();
        let t = [c.f64()?, c.f64()?, c.f64()?];
        let q = [c.f64()?, c.f64()?, c.f64()?, c.f64()?];
        out.push(Transform { parent: h.frame_id, child, t, q });
    }
    Ok(out)
}

/// geometry_msgs/PointStamped.
pub fn point_stamped(buf: &[u8]) -> Result<(Header, [f64; 3])> {
    let mut c = Cdr::new(buf)?;
    let h = header(&mut c)?;
    Ok((h, [c.f64()?, c.f64()?, c.f64()?]))
}

/// std_msgs/String (robot descriptions).
pub fn string(buf: &[u8]) -> Result<String> {
    let mut c = Cdr::new(buf)?;
    Ok(c.string()?.to_owned())
}

pub struct Marker {
    pub frame_id: String,
    pub ns: String,
    pub id: i32,
    pub kind: i32,
    pub action: i32,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub scale: [f64; 3],
    pub color: [f32; 4],
    pub lifetime: f64,
    pub points: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    /// JPEG (or other) bytes of the embedded texture, empty if none.
    pub texture: Vec<u8>,
    pub uv: Vec<[f32; 2]>,
    pub text: String,
    pub mesh_resource: String,
}

fn f32(c: &mut Cdr) -> Result<f32> {
    Ok(f32::from_bits(c.u32()?))
}

fn marker(c: &mut Cdr) -> Result<Marker> {
    let h = header(c)?;
    let ns = c.string()?.to_owned();
    let id = c.i32()?;
    let kind = c.i32()?;
    let action = c.i32()?;
    let (p, q) = pose(c)?;
    let scale = [c.f64()?, c.f64()?, c.f64()?];
    let color = [f32(c)?, f32(c)?, f32(c)?, f32(c)?];
    let lifetime = c.i32()? as f64 + c.u32()? as f64 * 1e-9;
    let _frame_locked = c.u8()?;
    let n = c.count(24)?;
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        points.push([c.f64()? as f32, c.f64()? as f32, c.f64()? as f32]);
    }
    let n = c.count(16)?;
    let mut colors = Vec::with_capacity(n);
    for _ in 0..n {
        colors.push([f32(c)?, f32(c)?, f32(c)?, f32(c)?]);
    }
    // ROS 1 (and ROS 2 before Humble) markers have no embedded texture, UV coordinates or mesh file.
    let (mut texture, mut uv) = (Vec::new(), Vec::new());
    if !c.ros1() {
        let _texture_resource = c.string()?;
        let _tex_header = header(c)?;
        let _tex_format = c.string()?;
        let (to, tl) = c.byte_seq()?;
        texture = c.slice(to, tl).to_vec();
        let n = c.count(8)?;
        uv.reserve(n);
        for _ in 0..n {
            uv.push([f32(c)?, f32(c)?]);
        }
    }
    let text = c.string()?.to_owned();
    let mesh_resource = c.string()?.to_owned();
    if !c.ros1() {
        let _mesh_file_name = c.string()?;
        let _mesh_file_data = c.byte_seq()?;
    }
    let _embedded = c.u8()?;
    Ok(Marker { frame_id: h.frame_id, ns, id, kind, action, pos: p, quat: q, scale, color, lifetime, points, colors, texture, uv, text, mesh_resource })
}

pub fn marker_msg(buf: &[u8]) -> Result<Vec<Marker>> {
    let mut c = Cdr::new(buf)?;
    Ok(vec![marker(&mut c)?])
}

pub fn marker_array(buf: &[u8]) -> Result<Vec<Marker>> {
    let mut c = Cdr::new(buf)?;
    let n = c.count(120)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(marker(&mut c)?);
    }
    Ok(out)
}
