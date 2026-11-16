//! Bag files, read without ROS: ROS 1 bags (.bag, format 2.0), ROS 2 MCAP (.mcap) and ROS 2 SQLite (.db3), or a
//! rosbag2 directory holding several of them.
//!
//! Opening builds one time-sorted index of every message (24 bytes per message) and nothing else. When the file
//! carries its own index (ROS 1 index records, MCAP summary) no chunk is decompressed for it; a file without index
//! (recording that was cut, unchunked MCAP) is scanned once. Messages are then read on demand, by entry.
//!
//! ROS 2 messages come out as recorded (CDR). ROS 1 messages come out with `cdr::ROS1_PREFIX` in front, which the
//! decoders in `msgs` understand.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::cdr::ROS1_PREFIX;

pub struct Topic {
    pub name: String,
    /// ROS 2 spelling ("sensor_msgs/msg/Image"), also for ROS 1 topics.
    pub type_name: String,
    pub ros1: bool,
    pub count: u64,
}

#[derive(Clone, Copy)]
pub struct Entry {
    /// Receive time, nanoseconds.
    pub t: u64,
    off: u64,
    chunk: u32,
    pub topic: u16,
    src: u16,
}

const NO_CHUNK: u32 = u32::MAX;

type Res<T> = Result<T, String>;

trait Source: Send {
    /// Appends the serialized message of `e` to `out`.
    fn read(&mut self, e: &Entry, out: &mut Vec<u8>) -> Res<()>;
}

struct Parsed {
    format: &'static str,
    topics: Vec<Topic>,
    entries: Vec<Entry>,
    source: Box<dyn Source>,
    warning: Option<String>,
}

pub struct Bag {
    pub format: String,
    pub topics: Vec<Topic>,
    /// Every message, sorted by time.
    pub entries: Vec<Entry>,
    pub warnings: Vec<String>,
    sources: Vec<Box<dyn Source>>,
}

impl Bag {
    /// Opens one bag made of these files (splits of one recording) and / or rosbag2 directories.
    pub fn open(paths: &[PathBuf]) -> Res<Bag> {
        let mut files = Vec::new();
        for p in paths {
            if p.is_dir() {
                let mut inside: Vec<PathBuf> = std::fs::read_dir(p)
                    .map_err(|e| format!("{}: {e}", p.display()))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|f| f.is_file() && matches!(f.extension().and_then(|x| x.to_str()), Some("mcap" | "db3" | "bag")))
                    .collect();
                inside.sort();
                if inside.is_empty() {
                    return Err(format!("{}: no .mcap, .db3 or .bag file in this directory", p.display()));
                }
                files.extend(inside);
            } else {
                files.push(p.clone());
            }
        }
        if files.is_empty() {
            return Err("no bag file given".into());
        }
        let mut bag = Bag { format: String::new(), topics: Vec::new(), entries: Vec::new(), warnings: Vec::new(), sources: Vec::new() };
        let mut by_name: HashMap<String, u16> = HashMap::new();
        for f in &files {
            let mut p = open_file(f).map_err(|e| format!("{}: {e}", f.display()))?;
            if let Some(w) = p.warning.take() {
                bag.warnings.push(format!("{}: {w}", f.file_name().unwrap_or_default().to_string_lossy()));
            }
            if !bag.format.contains(p.format) {
                if !bag.format.is_empty() {
                    bag.format.push_str(" + ");
                }
                bag.format.push_str(p.format);
            }
            let src = bag.sources.len();
            if src > u16::MAX as usize {
                return Err("too many files".into());
            }
            // Local topic index -> index in the merged list (same topic name in two splits = one topic).
            let mut remap = Vec::with_capacity(p.topics.len());
            for t in p.topics {
                let id = match by_name.get(&t.name) {
                    Some(id) => *id,
                    None => {
                        if bag.topics.len() >= u16::MAX as usize {
                            return Err("too many topics".into());
                        }
                        let id = bag.topics.len() as u16;
                        by_name.insert(t.name.clone(), id);
                        bag.topics.push(t);
                        id
                    }
                };
                remap.push(id);
            }
            bag.entries.reserve(p.entries.len());
            for mut e in p.entries {
                e.topic = remap[e.topic as usize];
                e.src = src as u16;
                bag.entries.push(e);
            }
            bag.sources.push(p.source);
        }
        bag.entries.sort_by_key(|e| e.t);
        for t in &mut bag.topics {
            t.count = 0;
        }
        for e in &bag.entries {
            bag.topics[e.topic as usize].count += 1;
        }
        Ok(bag)
    }

    /// The serialized message of an entry (ROS 1: behind `ROS1_PREFIX`).
    pub fn read(&mut self, e: &Entry) -> Res<Vec<u8>> {
        let mut out = Vec::new();
        if self.topics[e.topic as usize].ros1 {
            out.extend_from_slice(&ROS1_PREFIX);
        }
        self.sources[e.src as usize].read(e, &mut out)?;
        Ok(out)
    }

    pub fn start(&self) -> u64 {
        self.entries.first().map_or(0, |e| e.t)
    }

    pub fn end(&self) -> u64 {
        self.entries.last().map_or(0, |e| e.t)
    }
}

/// "sensor_msgs/Image" (ROS 1) -> "sensor_msgs/msg/Image"; ROS 2 names are kept.
fn ros2_type(t: &str) -> String {
    match t {
        "tf/tfMessage" => "tf2_msgs/msg/TFMessage".into(),
        _ => match t.split_once('/') {
            Some((pkg, name)) if !name.contains('/') => format!("{pkg}/msg/{name}"),
            _ => t.to_owned(),
        },
    }
}

fn open_file(path: &Path) -> Res<Parsed> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() < 16 {
        return Err("file too short to be a bag".into());
    }
    // Read-only map: the file is never written, and nothing is loaded until it is read.
    let map = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
    if map.starts_with(b"#ROSBAG V2.0\n") {
        open_ros1(map)
    } else if map.starts_with(&MCAP_MAGIC) {
        open_mcap(map)
    } else if map.starts_with(b"SQLite format 3\0") {
        drop(map);
        open_sqlite(path)
    } else if map.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Err("this file is zstd-compressed as a whole (rosbag2 file compression): decompress it first (zstd -d)".into())
    } else {
        Err("not a ROS 1 bag, an MCAP file or a rosbag2 SQLite file".into())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Byte readers and chunk decompression
// ---------------------------------------------------------------------------------------------------------------

/// Bounds-checked little-endian reader.
struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|e| *e <= self.b.len()).ok_or("truncated record")?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u16(&mut self) -> Res<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// u32 length, then that many bytes.
    fn bytes32(&mut self) -> Res<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    fn str32(&mut self) -> Res<&'a str> {
        std::str::from_utf8(self.bytes32()?).map_err(|_| "string is not UTF-8".to_string())
    }
    fn left(&self) -> usize {
        self.b.len() - self.p
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Comp {
    None,
    Lz4,
    Zstd,
    Bz2,
}

impl Comp {
    fn of(name: &str) -> Res<Comp> {
        Ok(match name {
            "" | "none" => Comp::None,
            "lz4" => Comp::Lz4,
            "zstd" => Comp::Zstd,
            "bz2" => Comp::Bz2,
            other => return Err(format!("unsupported chunk compression \"{other}\"")),
        })
    }
}

fn decompress(comp: Comp, src: &[u8], size: usize) -> Res<Vec<u8>> {
    // The declared size is only a hint: never trust it for more than a sane first allocation.
    let mut out = Vec::with_capacity(size.min(256 << 20));
    let r = match comp {
        Comp::None => {
            out.extend_from_slice(src);
            Ok(0)
        }
        Comp::Lz4 => lz4_flex::frame::FrameDecoder::new(src).read_to_end(&mut out),
        Comp::Zstd => zstd::stream::read::Decoder::new(src).and_then(|mut d| d.read_to_end(&mut out)),
        Comp::Bz2 => bzip2::read::BzDecoder::new(src).read_to_end(&mut out),
    };
    r.map_err(|e| format!("chunk decompression: {e}"))?;
    Ok(out)
}

/// Hands the pages of a mapped bag back to the system once `RELEASE_EVERY` bytes were read through it. They stay in
/// the page cache (a later read of the same place is still quick), but no longer count as memory of this process:
/// playing a 20 GB bag does not grow the process by 20 GB.
fn release_pages(map: &Mmap, read: &mut usize, n: usize) {
    const RELEASE_EVERY: usize = 256 << 20;
    *read += n;
    if *read >= RELEASE_EVERY {
        *read = 0;
        // Sound for a read-only file mapping: a dropped page is read again from the file on its next access.
        let _ = unsafe { map.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
    }
}

/// The few most recently decompressed chunks (messages close in time sit in the same chunk).
#[derive(Default)]
struct ChunkCache {
    slots: Vec<(u32, Vec<u8>)>,
}

impl ChunkCache {
    const KEEP: usize = 4;

    fn get(&mut self, id: u32, load: impl FnOnce() -> Res<Vec<u8>>) -> Res<&[u8]> {
        if let Some(i) = self.slots.iter().position(|s| s.0 == id) {
            let s = self.slots.remove(i);
            self.slots.push(s);
        } else {
            let data = load()?;
            if self.slots.len() >= Self::KEEP {
                self.slots.remove(0);
            }
            self.slots.push((id, data));
        }
        Ok(&self.slots.last().unwrap().1)
    }
}

/// Topics of one file while it is indexed: recorded connection / channel id -> local topic index.
#[derive(Default)]
struct TopicTable {
    topics: Vec<Topic>,
    by_name: HashMap<String, u16>,
    by_id: HashMap<u32, u16>,
}

impl TopicTable {
    fn add(&mut self, id: u32, name: &str, type_name: &str, ros1: bool) -> Res<()> {
        let idx = match self.by_name.get(name) {
            Some(i) => *i,
            None => {
                if self.topics.len() >= u16::MAX as usize {
                    return Err("too many topics".into());
                }
                let i = self.topics.len() as u16;
                self.topics.push(Topic { name: name.to_owned(), type_name: ros2_type(type_name), ros1, count: 0 });
                self.by_name.insert(name.to_owned(), i);
                i
            }
        };
        self.by_id.insert(id, idx);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ROS 1 bag, format 2.0
//
// "#ROSBAG V2.0\n", then records: u32 header length, header (fields: u32 length, "name=value"), u32 data length, data.
// op 3 bag header (index_pos), op 5 chunk (compression, size; data = records, maybe compressed), op 7 connection,
// op 2 message (conn, time), op 4 index of one connection in the preceding chunk, op 6 chunk info (chunk_pos).
// ---------------------------------------------------------------------------------------------------------------

struct R1Rec<'a> {
    header: &'a [u8],
    data_off: usize,
    data_len: usize,
    next: usize,
}

fn r1_record(b: &[u8], pos: usize) -> Res<R1Rec<'_>> {
    let mut r = Rd { b, p: pos };
    let header = r.bytes32()?;
    let data_len = r.u32()? as usize;
    let data_off = r.p;
    r.take(data_len)?;
    Ok(R1Rec { header, data_off, data_len, next: r.p })
}

fn r1_field<'a>(header: &'a [u8], name: &str) -> Option<&'a [u8]> {
    let mut r = Rd::new(header);
    while r.left() >= 4 {
        let f = r.bytes32().ok()?;
        let eq = f.iter().position(|c| *c == b'=')?;
        if &f[..eq] == name.as_bytes() {
            return Some(&f[eq + 1..]);
        }
    }
    None
}

fn r1_u32(header: &[u8], name: &str) -> Res<u32> {
    r1_field(header, name).and_then(|v| v.try_into().ok()).map(u32::from_le_bytes).ok_or_else(|| format!("record without \"{name}\""))
}

fn r1_u64(header: &[u8], name: &str) -> Res<u64> {
    r1_field(header, name).and_then(|v| v.try_into().ok()).map(u64::from_le_bytes).ok_or_else(|| format!("record without \"{name}\""))
}

fn r1_op(header: &[u8]) -> u8 {
    r1_field(header, "op").and_then(|v| v.first().copied()).unwrap_or(0)
}

/// ROS 1 time (u32 seconds, u32 nanoseconds) to nanoseconds.
fn r1_time(b: &[u8]) -> u64 {
    u32::from_le_bytes(b[0..4].try_into().unwrap()) as u64 * 1_000_000_000 + u32::from_le_bytes(b[4..8].try_into().unwrap()) as u64
}

fn r1_connection(table: &mut TopicTable, b: &[u8], rec: &R1Rec) -> Res<()> {
    let conn = r1_u32(rec.header, "conn")?;
    let topic = r1_field(rec.header, "topic").ok_or("connection without topic")?;
    let data = &b[rec.data_off..rec.data_off + rec.data_len];
    let ty = r1_field(data, "type").ok_or("connection without type")?;
    table.add(conn, &String::from_utf8_lossy(topic), &String::from_utf8_lossy(ty), true)
}

struct R1Chunk {
    off: usize,
    len: usize,
    size: usize,
    comp: Comp,
}

struct Ros1 {
    map: Mmap,
    chunks: Vec<R1Chunk>,
    cache: ChunkCache,
    read: usize,
}

fn r1_chunk(b: &[u8], pos: usize) -> Res<(R1Chunk, usize)> {
    let rec = r1_record(b, pos)?;
    if r1_op(rec.header) != 5 {
        return Err("chunk record expected".into());
    }
    let comp = Comp::of(&String::from_utf8_lossy(r1_field(rec.header, "compression").unwrap_or(b"none")))?;
    let size = r1_u32(rec.header, "size")? as usize;
    Ok((R1Chunk { off: rec.data_off, len: rec.data_len, size, comp }, rec.next))
}

/// Index from the bag's own index section: connections and chunk infos at `index_pos`, then per chunk the index
/// records that follow it. No chunk is decompressed.
fn r1_indexed(b: &[u8], index_pos: usize, table: &mut TopicTable, chunks: &mut Vec<R1Chunk>, entries: &mut Vec<Entry>) -> Res<()> {
    let mut chunk_pos = Vec::new();
    let mut pos = index_pos;
    while pos < b.len() {
        let rec = r1_record(b, pos)?;
        match r1_op(rec.header) {
            7 => r1_connection(table, b, &rec)?,
            6 => chunk_pos.push(r1_u64(rec.header, "chunk_pos")? as usize),
            _ => {}
        }
        pos = rec.next;
    }
    chunk_pos.sort_unstable();
    for cp in chunk_pos {
        let (chunk, mut pos) = r1_chunk(b, cp)?;
        let id = chunks.len() as u32;
        chunks.push(chunk);
        while pos < b.len() {
            let rec = r1_record(b, pos)?;
            if r1_op(rec.header) != 4 {
                break;
            }
            pos = rec.next;
            let Some(topic) = table.by_id.get(&r1_u32(rec.header, "conn")?).copied() else { continue };
            let data = &b[rec.data_off..rec.data_off + rec.data_len];
            for e in data.chunks_exact(12) {
                entries.push(Entry { t: r1_time(e), off: u32::from_le_bytes(e[8..12].try_into().unwrap()) as u64, chunk: id, topic, src: 0 });
            }
        }
    }
    Ok(())
}

/// Index by reading every record of the file (no index section: a recording that was not closed).
fn r1_scan(b: &[u8], mut pos: usize, table: &mut TopicTable, chunks: &mut Vec<R1Chunk>, entries: &mut Vec<Entry>) -> Res<()> {
    while pos < b.len() {
        let rec = r1_record(b, pos)?;
        match r1_op(rec.header) {
            7 => r1_connection(table, b, &rec)?,
            5 => {
                let (chunk, _) = r1_chunk(b, pos)?;
                let id = chunks.len() as u32;
                let owned;
                let data: &[u8] = if chunk.comp == Comp::None {
                    &b[chunk.off..chunk.off + chunk.len]
                } else {
                    owned = decompress(chunk.comp, &b[chunk.off..chunk.off + chunk.len], chunk.size)?;
                    &owned
                };
                chunks.push(chunk);
                let mut ip = 0;
                while ip < data.len() {
                    let Ok(inner) = r1_record(data, ip) else { break };
                    match r1_op(inner.header) {
                        7 => r1_connection(table, data, &inner)?,
                        2 => {
                            let conn = r1_u32(inner.header, "conn")?;
                            let time = r1_field(inner.header, "time").filter(|t| t.len() == 8).ok_or("message without time")?;
                            if let Some(topic) = table.by_id.get(&conn).copied() {
                                entries.push(Entry { t: r1_time(time), off: ip as u64, chunk: id, topic, src: 0 });
                            }
                        }
                        _ => {}
                    }
                    ip = inner.next;
                }
            }
            _ => {}
        }
        pos = rec.next;
    }
    Ok(())
}

fn open_ros1(map: Mmap) -> Res<Parsed> {
    let b = &map[..];
    let head = r1_record(b, 13)?;
    if r1_op(head.header) != 3 {
        return Err("ROS 1 bag without bag header record".into());
    }
    let index_pos = r1_u64(head.header, "index_pos").unwrap_or(0) as usize;
    let (mut table, mut chunks, mut entries) = (TopicTable::default(), Vec::new(), Vec::new());
    let mut warning = None;
    let indexed = index_pos > 0 && index_pos < b.len() && r1_indexed(b, index_pos, &mut table, &mut chunks, &mut entries).is_ok() && !entries.is_empty();
    if !indexed {
        (table, chunks, entries) = (TopicTable::default(), Vec::new(), Vec::new());
        if let Err(e) = r1_scan(b, head.next, &mut table, &mut chunks, &mut entries) {
            warning = Some(format!("unindexed bag, stopped at a damaged record ({e}): {} messages recovered", entries.len()));
        } else {
            warning = Some("unindexed bag (recording not closed?): scanned in full".into());
        }
    }
    Ok(Parsed { format: "ROS 1 bag", topics: table.topics, entries, source: Box::new(Ros1 { map, chunks, cache: ChunkCache::default(), read: usize::MAX / 2 }), warning })
}

impl Source for Ros1 {
    fn read(&mut self, e: &Entry, out: &mut Vec<u8>) -> Res<()> {
        let c = self.chunks.get(e.chunk as usize).ok_or("bad chunk index")?;
        let raw = &self.map[c.off..c.off + c.len];
        let data: &[u8] = if c.comp == Comp::None { raw } else { self.cache.get(e.chunk, || decompress(c.comp, raw, c.size))? };
        let rec = r1_record(data, e.off as usize)?;
        if r1_op(rec.header) != 2 {
            return Err("index does not point at a message".into());
        }
        out.extend_from_slice(&data[rec.data_off..rec.data_off + rec.data_len]);
        release_pages(&self.map, &mut self.read, rec.data_len);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// MCAP
//
// Magic, records (u8 opcode, u64 length, content), magic. 0x03 schema, 0x04 channel, 0x05 message, 0x06 chunk (records,
// maybe compressed), 0x07 message index of one channel in the preceding chunk, 0x08 chunk index, 0x02 footer (where the
// summary section starts).
// ---------------------------------------------------------------------------------------------------------------

const MCAP_MAGIC: [u8; 8] = [0x89, b'M', b'C', b'A', b'P', b'0', b'\r', b'\n'];
const MCAP_FOOTER: usize = 1 + 8 + 20;

struct McRec {
    op: u8,
    off: usize,
    len: usize,
    next: usize,
}

fn mc_record(b: &[u8], pos: usize) -> Res<McRec> {
    let mut r = Rd { b, p: pos };
    let op = r.take(1)?[0];
    let len = usize::try_from(r.u64()?).map_err(|_| "record too long")?;
    let off = r.p;
    r.take(len)?;
    Ok(McRec { op, off, len, next: r.p })
}

struct McChunk {
    off: usize,
    len: usize,
    size: usize,
    comp: Comp,
}

/// Parses a chunk record's content: where its (maybe compressed) records are.
fn mc_chunk(b: &[u8], rec: &McRec) -> Res<McChunk> {
    let mut r = Rd { b: &b[..rec.off + rec.len], p: rec.off };
    r.u64()?;
    r.u64()?;
    let size = r.u64()? as usize;
    r.u32()?;
    let comp = Comp::of(r.str32()?)?;
    let len = r.u64()? as usize;
    let off = r.p;
    r.take(len)?;
    Ok(McChunk { off, len, size, comp })
}

#[derive(Default)]
struct McTopics {
    table: TopicTable,
    schemas: HashMap<u16, String>,
}

impl McTopics {
    fn schema(&mut self, content: &[u8]) -> Res<()> {
        let mut r = Rd::new(content);
        let id = r.u16()?;
        self.schemas.insert(id, r.str32()?.to_owned());
        Ok(())
    }
    fn channel(&mut self, content: &[u8]) -> Res<()> {
        let mut r = Rd::new(content);
        let id = r.u16()?;
        let schema = r.u16()?;
        let topic = r.str32()?;
        let encoding = r.str32()?;
        let ty = self.schemas.get(&schema).cloned().unwrap_or_default();
        self.table.add(id as u32, topic, &ty, encoding == "ros1")
    }
}

/// Messages of one chunk, by reading its records (used when the file has no message index).
fn mc_walk_chunk(data: &[u8], id: u32, topics: &mut McTopics, entries: &mut Vec<Entry>) -> Res<()> {
    let mut pos = 0;
    while pos < data.len() {
        let rec = mc_record(data, pos)?;
        let content = &data[rec.off..rec.off + rec.len];
        match rec.op {
            0x03 => topics.schema(content)?,
            0x04 => topics.channel(content)?,
            0x05 => mc_message(content, pos, id, topics, entries)?,
            _ => {}
        }
        pos = rec.next;
    }
    Ok(())
}

fn mc_message(content: &[u8], rec_pos: usize, chunk: u32, topics: &McTopics, entries: &mut Vec<Entry>) -> Res<()> {
    let mut r = Rd::new(content);
    let channel = r.u16()?;
    r.u32()?;
    let t = r.u64()?;
    if let Some(topic) = topics.table.by_id.get(&(channel as u32)).copied() {
        entries.push(Entry { t, off: rec_pos as u64, chunk, topic, src: 0 });
    }
    Ok(())
}

fn mc_chunk_data<'a>(b: &'a [u8], c: &McChunk, owned: &'a mut Vec<u8>) -> Res<&'a [u8]> {
    let raw = &b[c.off..c.off + c.len];
    if c.comp == Comp::None {
        return Ok(raw);
    }
    *owned = decompress(c.comp, raw, c.size)?;
    Ok(owned)
}

/// Index from the summary section: schemas, channels and chunk indexes, then the message index records that follow
/// each chunk. No chunk is decompressed (unless the writer left the message indexes out).
fn mc_indexed(b: &[u8], topics: &mut McTopics, chunks: &mut Vec<McChunk>, entries: &mut Vec<Entry>) -> Res<()> {
    let footer_at = b.len().checked_sub(MCAP_MAGIC.len() + MCAP_FOOTER).ok_or("no footer")?;
    if b[b.len() - MCAP_MAGIC.len()..] != MCAP_MAGIC {
        return Err("no closing magic".into());
    }
    let footer = mc_record(b, footer_at)?;
    if footer.op != 0x02 {
        return Err("no footer".into());
    }
    let summary_start = Rd { b, p: footer.off }.u64()? as usize;
    if summary_start == 0 || summary_start >= footer_at {
        return Err("no summary".into());
    }
    // Schemas first (a channel names its schema), then channels and chunk indexes.
    let mut recs = Vec::new();
    let mut pos = summary_start;
    while pos < footer_at {
        let rec = mc_record(b, pos)?;
        pos = rec.next;
        if rec.op == 0x03 {
            topics.schema(&b[rec.off..rec.off + rec.len])?;
        } else if matches!(rec.op, 0x04 | 0x08) {
            recs.push(rec);
        }
    }
    for rec in recs.iter().filter(|r| r.op == 0x04) {
        topics.channel(&b[rec.off..rec.off + rec.len])?;
    }
    for rec in recs.iter().filter(|r| r.op == 0x08) {
        let mut r = Rd::new(&b[rec.off..rec.off + rec.len]);
        r.u64()?;
        r.u64()?;
        let chunk_start = r.u64()? as usize;
        let chunk_len = r.u64()? as usize;
        r.bytes32()?;
        let index_len = r.u64()? as usize;
        let chunk_rec = mc_record(b, chunk_start)?;
        if chunk_rec.op != 0x06 {
            return Err("chunk index does not point at a chunk".into());
        }
        let chunk = mc_chunk(b, &chunk_rec)?;
        let id = chunks.len() as u32;
        if index_len == 0 {
            let mut owned = Vec::new();
            mc_walk_chunk(mc_chunk_data(b, &chunk, &mut owned)?, id, topics, entries)?;
        } else {
            let (mut pos, end) = (chunk_start + chunk_len, chunk_start + chunk_len + index_len);
            while pos < end {
                let ix = mc_record(b, pos)?;
                pos = ix.next;
                if ix.op != 0x07 {
                    continue;
                }
                let mut r = Rd::new(&b[ix.off..ix.off + ix.len]);
                let channel = r.u16()?;
                let Some(topic) = topics.table.by_id.get(&(channel as u32)).copied() else { continue };
                for e in r.bytes32()?.chunks_exact(16) {
                    let t = u64::from_le_bytes(e[0..8].try_into().unwrap());
                    let off = u64::from_le_bytes(e[8..16].try_into().unwrap());
                    entries.push(Entry { t, off, chunk: id, topic, src: 0 });
                }
            }
        }
        chunks.push(chunk);
    }
    Ok(())
}

/// Index by reading every record of the data section (no summary: unfinished or unindexed file).
fn mc_scan(b: &[u8], topics: &mut McTopics, chunks: &mut Vec<McChunk>, entries: &mut Vec<Entry>) -> Res<()> {
    let mut pos = MCAP_MAGIC.len();
    while pos < b.len() {
        let rec = mc_record(b, pos)?;
        let content = &b[rec.off..rec.off + rec.len];
        match rec.op {
            0x03 => topics.schema(content)?,
            0x04 => topics.channel(content)?,
            0x05 => mc_message(content, pos, NO_CHUNK, topics, entries)?,
            0x06 => {
                let chunk = mc_chunk(b, &rec)?;
                let mut owned = Vec::new();
                mc_walk_chunk(mc_chunk_data(b, &chunk, &mut owned)?, chunks.len() as u32, topics, entries)?;
                chunks.push(chunk);
            }
            0x02 | 0x0f => break, // footer / data end: the summary only repeats what was read
            _ => {}
        }
        pos = rec.next;
    }
    Ok(())
}

struct Mcap {
    map: Mmap,
    chunks: Vec<McChunk>,
    cache: ChunkCache,
    read: usize,
}

fn open_mcap(map: Mmap) -> Res<Parsed> {
    let b = &map[..];
    let (mut topics, mut chunks, mut entries) = (McTopics::default(), Vec::new(), Vec::new());
    let mut warning = None;
    let indexed = mc_indexed(b, &mut topics, &mut chunks, &mut entries).is_ok() && !entries.is_empty();
    if !indexed {
        (topics, chunks, entries) = (McTopics::default(), Vec::new(), Vec::new());
        if let Err(e) = mc_scan(b, &mut topics, &mut chunks, &mut entries) {
            warning = Some(format!("no index, stopped at a damaged record ({e}): {} messages recovered", entries.len()));
        }
    }
    Ok(Parsed { format: "MCAP", topics: topics.table.topics, entries, source: Box::new(Mcap { map, chunks, cache: ChunkCache::default(), read: usize::MAX / 2 }), warning })
}

impl Source for Mcap {
    fn read(&mut self, e: &Entry, out: &mut Vec<u8>) -> Res<()> {
        let data: &[u8] = if e.chunk == NO_CHUNK {
            &self.map[..]
        } else {
            let c = self.chunks.get(e.chunk as usize).ok_or("bad chunk index")?;
            let raw = &self.map[c.off..c.off + c.len];
            if c.comp == Comp::None { raw } else { self.cache.get(e.chunk, || decompress(c.comp, raw, c.size))? }
        };
        let rec = mc_record(data, e.off as usize)?;
        // channel u16, sequence u32, log time u64, publish time u64, then the message.
        if rec.op != 0x05 || rec.len < 22 {
            return Err("index does not point at a message".into());
        }
        out.extend_from_slice(&data[rec.off + 22..rec.off + rec.len]);
        release_pages(&self.map, &mut self.read, rec.len);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// rosbag2 SQLite (.db3): topics(id, name, type, ...), messages(id, topic_id, timestamp, data)
// ---------------------------------------------------------------------------------------------------------------

struct Sqlite {
    conn: rusqlite::Connection,
}

fn open_sqlite(path: &Path) -> Res<Parsed> {
    use rusqlite::OpenFlags;
    let e = |e: rusqlite::Error| e.to_string();
    let conn = rusqlite::Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).map_err(e)?;
    let mut table = TopicTable::default();
    {
        let mut st = conn.prepare("SELECT id, name, type FROM topics").map_err(|_| "SQLite file without a rosbag2 topics table".to_string())?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))).map_err(e)?;
        for row in rows {
            let (id, name, ty) = row.map_err(e)?;
            table.add(id as u32, &name, &ty, false)?;
        }
    }
    let mut entries = Vec::new();
    {
        let mut st = conn.prepare("SELECT id, topic_id, timestamp FROM messages").map_err(e)?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))).map_err(e)?;
        for row in rows {
            let (id, topic, t) = row.map_err(e)?;
            if let Some(topic) = table.by_id.get(&(topic as u32)).copied() {
                entries.push(Entry { t: t.max(0) as u64, off: id as u64, chunk: NO_CHUNK, topic, src: 0 });
            }
        }
    }
    Ok(Parsed { format: "SQLite", topics: table.topics, entries, source: Box::new(Sqlite { conn }), warning: None })
}

impl Source for Sqlite {
    fn read(&mut self, e: &Entry, out: &mut Vec<u8>) -> Res<()> {
        let mut st = self.conn.prepare_cached("SELECT data FROM messages WHERE id = ?1").map_err(|e| e.to_string())?;
        let mut rows = st.query([e.off as i64]).map_err(|e| e.to_string())?;
        let row = rows.next().map_err(|e| e.to_string())?.ok_or("message not found")?;
        let data = row.get_ref(0).map_err(|e| e.to_string())?.as_blob().map_err(|e| e.to_string())?;
        out.extend_from_slice(data);
        Ok(())
    }
}
