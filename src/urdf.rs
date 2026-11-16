//! Robot model from a URDF string (/robot_description): visuals of every link (box, cylinder, sphere, STL or COLLADA
//! mesh), placed each frame by the TF of their link. `package://pkg/...` resolves through AMENT_PREFIX_PATH.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use glam::{DAffine3, DQuat, DVec3, Vec3};

use crate::scene::{MeshVertex, RobotVisual};
use crate::shapes;

fn nums(s: &str) -> Vec<f64> {
    s.split_whitespace().filter_map(|x| x.parse().ok()).collect()
}

fn resolve(uri: &str) -> Option<PathBuf> {
    if let Some(rest) = uri.strip_prefix("package://") {
        let (pkg, rel) = rest.split_once('/')?;
        let prefixes = std::env::var("AMENT_PREFIX_PATH").unwrap_or_default();
        return prefixes.split(':').map(|p| PathBuf::from(p).join("share").join(pkg).join(rel)).find(|p| p.exists());
    }
    let p = PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri));
    if p.exists() {
        return Some(p);
    }
    // A path of another machine or container: look for the same file name in RSVIZ_MESH_PATH (colon-separated dirs).
    let name = p.file_name()?;
    std::env::var("RSVIZ_MESH_PATH").unwrap_or_default().split(':').filter(|d| !d.is_empty()).map(|d| PathBuf::from(d).join(name)).find(|q| q.exists())
}

fn load_stl(bytes: &[u8], col: [f32; 4], scale: Vec3) -> Vec<MeshVertex> {
    let mut out = Vec::new();
    let push = |out: &mut Vec<MeshVertex>, p: [f32; 3]| out.push(MeshVertex { pos: (Vec3::from(p) * scale).to_array(), color: col, uv: [0.0; 2] });
    if bytes.len() >= 84 {
        let n = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        if 84 + n * 50 == bytes.len() {
            for t in 0..n {
                let b = 84 + t * 50 + 12;
                for k in 0..3 {
                    let o = b + k * 12;
                    let f = |i: usize| f32::from_le_bytes(bytes[o + 4 * i..o + 4 * i + 4].try_into().unwrap());
                    push(&mut out, [f(0), f(1), f(2)]);
                }
            }
            return out;
        }
    }
    // ASCII STL
    for line in String::from_utf8_lossy(bytes).lines() {
        let l = line.trim();
        if let Some(r) = l.strip_prefix("vertex") {
            let v = nums(r);
            if v.len() == 3 {
                push(&mut out, [v[0] as f32, v[1] as f32, v[2] as f32]);
            }
        }
    }
    out.truncate(out.len() / 3 * 3);
    out
}

/// COLLADA <triangles> with their material's diffuse colour (enough for exported CAD / generated models).
fn load_dae(text: &str, fallback: [f32; 4], scale: Vec3) -> Vec<MeshVertex> {
    let Ok(doc) = roxmltree::Document::parse(text) else { return Vec::new() };
    let by_id: HashMap<&str, roxmltree::Node> = doc.descendants().filter_map(|n| n.attribute("id").map(|id| (id, n))).collect();
    let strip = |s: &str| s.trim_start_matches('#').to_owned();
    // material id -> diffuse colour
    let mut mat_color: HashMap<String, [f32; 4]> = HashMap::new();
    for m in doc.descendants().filter(|n| n.has_tag_name("material")) {
        let Some(id) = m.attribute("id") else { continue };
        let eff = m.children().find(|c| c.has_tag_name("instance_effect")).and_then(|c| c.attribute("url")).map(strip);
        if let Some(e) = eff.and_then(|e| by_id.get(e.as_str()).copied()) {
            if let Some(c) = e.descendants().find(|n| n.has_tag_name("diffuse")).and_then(|d| d.children().find(|c| c.has_tag_name("color"))).and_then(|c| c.text()) {
                let v = nums(c);
                if v.len() >= 3 {
                    mat_color.insert(id.to_owned(), [v[0] as f32, v[1] as f32, v[2] as f32, v.get(3).copied().unwrap_or(1.0) as f32]);
                }
            }
        }
    }
    // symbol -> material id, from the scene's bind_material
    let symbol: HashMap<String, String> = doc
        .descendants()
        .filter(|n| n.has_tag_name("instance_material"))
        .filter_map(|n| Some((n.attribute("symbol")?.to_owned(), strip(n.attribute("target")?))))
        .collect();
    let mut out = Vec::new();
    for tri in doc.descendants().filter(|n| n.has_tag_name("triangles")) {
        let mesh = tri.parent().unwrap();
        let inputs: Vec<roxmltree::Node> = tri.children().filter(|c| c.has_tag_name("input")).collect();
        let stride = inputs.iter().filter_map(|i| i.attribute("offset")?.parse::<usize>().ok()).max().unwrap_or(0) + 1;
        let Some(vin) = inputs.iter().find(|i| i.attribute("semantic") == Some("VERTEX")) else { continue };
        let voff: usize = vin.attribute("offset").and_then(|o| o.parse().ok()).unwrap_or(0);
        let vid = strip(vin.attribute("source").unwrap_or(""));
        let pos_src = mesh
            .children()
            .find(|c| c.has_tag_name("vertices") && c.attribute("id") == Some(vid.as_str()))
            .and_then(|v| v.children().find(|i| i.attribute("semantic") == Some("POSITION")))
            .and_then(|i| i.attribute("source"))
            .map(strip);
        let Some(src) = pos_src.and_then(|s| by_id.get(s.as_str()).copied()) else { continue };
        let Some(fa) = src.descendants().find(|n| n.has_tag_name("float_array")).and_then(|n| n.text()) else { continue };
        let pos = nums(fa);
        let col = tri.attribute("material").and_then(|s| symbol.get(s)).and_then(|m| mat_color.get(m)).copied().unwrap_or(fallback);
        let Some(p) = tri.children().find(|c| c.has_tag_name("p")).and_then(|n| n.text()) else { continue };
        let idx: Vec<usize> = p.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        for k in (0..idx.len()).step_by(stride) {
            let i = idx[k + voff] * 3;
            if i + 2 < pos.len() {
                out.push(MeshVertex { pos: (Vec3::new(pos[i] as f32, pos[i + 1] as f32, pos[i + 2] as f32) * scale).to_array(), color: col, uv: [0.0; 2] });
            }
        }
    }
    out.truncate(out.len() / 3 * 3);
    out
}

/// Visuals of a URDF, and the meshes that could not be loaded.
pub fn parse(urdf: &str) -> Result<(Vec<RobotVisual>, Vec<String>), String> {
    let doc = roxmltree::Document::parse(urdf).map_err(|e| format!("URDF: {e}"))?;
    let robot = doc.root_element();
    let mut named: HashMap<String, [f32; 4]> = HashMap::new();
    let rgba_of = |m: roxmltree::Node| m.children().find(|c| c.has_tag_name("color")).and_then(|c| c.attribute("rgba")).map(nums).filter(|v| v.len() == 4).map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]);
    for m in robot.children().filter(|c| c.has_tag_name("material")) {
        if let (Some(n), Some(c)) = (m.attribute("name"), rgba_of(m)) {
            named.insert(n.to_owned(), c);
        }
    }
    let (mut visuals, mut missing) = (Vec::new(), Vec::new());
    for link in robot.children().filter(|c| c.has_tag_name("link")) {
        let lname = link.attribute("name").unwrap_or("").to_owned();
        for vis in link.children().filter(|c| c.has_tag_name("visual")) {
            let origin = vis.children().find(|c| c.has_tag_name("origin"));
            let xyz = origin.and_then(|o| o.attribute("xyz")).map(nums).filter(|v| v.len() == 3).unwrap_or(vec![0.0; 3]);
            let rpy = origin.and_then(|o| o.attribute("rpy")).map(nums).filter(|v| v.len() == 3).unwrap_or(vec![0.0; 3]);
            let o = DAffine3::from_rotation_translation(DQuat::from_euler(glam::EulerRot::ZYX, rpy[2], rpy[1], rpy[0]), DVec3::new(xyz[0], xyz[1], xyz[2]));
            let col = vis
                .children()
                .find(|c| c.has_tag_name("material"))
                .and_then(|m| rgba_of(m).or_else(|| m.attribute("name").and_then(|n| named.get(n).copied())))
                .unwrap_or([0.7, 0.7, 0.72, 1.0]);
            let Some(geo) = vis.children().find(|c| c.has_tag_name("geometry")).and_then(|g| g.children().find(|c| c.is_element())) else { continue };
            let mut verts = Vec::new();
            match geo.tag_name().name() {
                "box" => {
                    let s = nums(geo.attribute("size").unwrap_or("0 0 0"));
                    if s.len() == 3 {
                        shapes::cube(&mut verts, Vec3::ZERO, Vec3::new(s[0] as f32, s[1] as f32, s[2] as f32), col);
                    }
                }
                "cylinder" => {
                    let r: f32 = geo.attribute("radius").and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    let l: f32 = geo.attribute("length").and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    shapes::frustum(&mut verts, Vec3::new(0., 0., -l / 2.), Vec3::new(0., 0., l / 2.), r, r, col);
                }
                "sphere" => {
                    let r: f32 = geo.attribute("radius").and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    shapes::sphere(&mut verts, Vec3::ZERO, Vec3::splat(2.0 * r), col);
                }
                "mesh" => {
                    let uri = geo.attribute("filename").unwrap_or("");
                    let s = geo.attribute("scale").map(nums).filter(|v| v.len() == 3).map(|v| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)).unwrap_or(Vec3::ONE);
                    match resolve(uri).and_then(|p| std::fs::read(&p).ok().map(|b| (p, b))) {
                        Some((p, b)) => {
                            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                            verts = match ext.as_str() {
                                "stl" => load_stl(&b, col, s),
                                "dae" => load_dae(&String::from_utf8_lossy(&b), col, s),
                                _ => {
                                    missing.push(format!("{uri} (format not supported)"));
                                    Vec::new()
                                }
                            };
                        }
                        None => missing.push(uri.to_owned()),
                    }
                }
                _ => {}
            }
            if !verts.is_empty() {
                visuals.push(RobotVisual { link: lname.clone(), origin: o, verts: Arc::new(verts) });
            }
        }
    }
    Ok((visuals, missing))
}
