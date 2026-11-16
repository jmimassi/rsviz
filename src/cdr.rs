//! Minimal reader for ROS 2 serialized messages (OMG CDR, XCDR1 as written by rmw_fastrtps / cyclonedds / zenoh).
//!
//! Layout: a 4-byte encapsulation header (0x00 0x00 = big endian, 0x00 0x01 = little endian, two option bytes), then the
//! body. Primitives are aligned to their own size, counted from the start of the body. Strings are a u32 length that
//! includes the trailing NUL, then the bytes. Sequences are a u32 element count, then the elements.
//!
//! The same reader also walks ROS 1 serialized messages (from ROS 1 bags): little endian, no alignment, no trailing NUL
//! in strings. Such a buffer is marked by the 4-byte prefix `ROS1_PREFIX` in place of the encapsulation header, so
//! every offset into the buffer keeps the same meaning in both encodings.

#[derive(Debug)]
pub struct CdrError(pub &'static str);

pub type Result<T> = std::result::Result<T, CdrError>;

pub struct Cdr<'a> {
    buf: &'a [u8],
    /// Absolute read position in `buf`.
    pos: usize,
    le: bool,
    ros1: bool,
}

const BODY: usize = 4;

/// Stands in for the CDR encapsulation header in front of a ROS 1 serialized message.
pub const ROS1_PREFIX: [u8; 4] = *b"R1\0\0";

impl<'a> Cdr<'a> {
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < BODY {
            return Err(CdrError("message shorter than the CDR header"));
        }
        if buf[..2] == ROS1_PREFIX[..2] {
            return Ok(Self { buf, pos: BODY, le: true, ros1: true });
        }
        let le = match buf[1] {
            0x01 | 0x03 => true,
            0x00 | 0x02 => false,
            _ => return Err(CdrError("unknown CDR encapsulation")),
        };
        Ok(Self { buf, pos: BODY, le, ros1: false })
    }

    /// True for a ROS 1 message (some messages have a different layout there, e.g. the `seq` of std_msgs/Header).
    pub fn ros1(&self) -> bool {
        self.ros1
    }

    fn align(&mut self, n: usize) {
        if self.ros1 {
            return;
        }
        let rel = self.pos - BODY;
        let pad = (n - rel % n) % n;
        self.pos += pad;
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(CdrError("length overflow"))?;
        if end > self.buf.len() {
            return Err(CdrError("message truncated"));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> Result<u32> {
        self.align(4);
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    }

    /// Element count of a sequence whose elements take at least `min` bytes each: a count that cannot fit in the rest
    /// of the message (corrupt data, wrong type) is an error instead of a huge allocation.
    pub fn count(&mut self, min: usize) -> Result<usize> {
        let n = self.u32()? as usize;
        if n.saturating_mul(min.max(1)) > self.buf.len() - self.pos {
            return Err(CdrError("sequence longer than the message"));
        }
        Ok(n)
    }

    pub fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    pub fn f64(&mut self) -> Result<f64> {
        self.align(8);
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.le { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) })
    }

    pub fn string(&mut self) -> Result<&'a str> {
        let n = self.u32()? as usize;
        let s = self.take(n)?;
        let s = s.strip_suffix(&[0]).unwrap_or(s);
        std::str::from_utf8(s).map_err(|_| CdrError("string is not UTF-8"))
    }

    /// Bytes of the original buffer at an absolute offset (as returned by `byte_seq`).
    pub fn slice(&self, off: usize, len: usize) -> &'a [u8] {
        &self.buf[off..off + len]
    }

    /// A `uint8[]` sequence: returns (absolute offset, length) of its bytes in the original buffer, without copying.
    pub fn byte_seq(&mut self) -> Result<(usize, usize)> {
        let n = self.u32()? as usize;
        let off = self.pos;
        self.take(n)?;
        Ok((off, n))
    }
}
