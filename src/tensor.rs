/// Dequantize INT8 affine data to f32.
///
/// Formula: `value = (raw - zp) * scale`
pub fn dequantize_affine(data: &[i8], zp: i32, scale: f32) -> Vec<f32> {
    data.iter()
        .map(|&v| (v as f32 - zp as f32) * scale)
        .collect()
}

/// Convert NC1HWC2 tensor layout to flat NCHW order.
///
/// RKNN NPU packs channels into blocks of `c2` (typically 16).
/// Input shape: `[1, c1, h, w, c2]` where `c1 * c2 >= total_channels`.
/// Returns a flat `[total_channels * h * w]` vector in NCHW order.
///
/// Works with any element type (i8 for raw output, f32 for dequantized).
pub fn nc1hwc2_to_flat<T: Copy + Default>(
    data: &[T],
    c1: usize,
    h: usize,
    w: usize,
    c2: usize,
    total_channels: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); total_channels * h * w];

    for c1_idx in 0..c1 {
        for y in 0..h {
            for x in 0..w {
                for c2_idx in 0..c2 {
                    let ch = c1_idx * c2 + c2_idx;
                    if ch >= total_channels {
                        continue;
                    }
                    let src_offset =
                        ((c1_idx * h + y) * w + x) * c2 + c2_idx;
                    let dst_offset = ch * h * w + y * w + x;
                    if src_offset < data.len() {
                        out[dst_offset] = data[src_offset];
                    }
                }
            }
        }
    }

    out
}

/// Metadata for a single output tensor.
#[derive(Debug, Clone)]
pub struct TensorAttr {
    pub index: u32,
    pub shape: Vec<u32>,
    pub n_elems: u32,
    pub size: u32,
    pub size_with_stride: u32,
    pub format: TensorFormat,
    pub data_type: TensorType,
    pub qnt_type: QuantType,
    pub zp: i32,
    pub scale: f32,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFormat {
    NCHW,
    NHWC,
    NC1HWC2,
    Undefined,
}

impl From<u32> for TensorFormat {
    fn from(v: u32) -> Self {
        match v {
            0 => TensorFormat::NCHW,
            1 => TensorFormat::NHWC,
            2 => TensorFormat::NC1HWC2,
            _ => TensorFormat::Undefined,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    Float32,
    Float16,
    Int8,
    Uint8,
    Int16,
    Int32,
    Unknown(u32),
}

impl From<u32> for TensorType {
    fn from(v: u32) -> Self {
        match v {
            0 => TensorType::Float32,
            1 => TensorType::Float16,
            2 => TensorType::Int8,
            3 => TensorType::Uint8,
            4 => TensorType::Int16,
            5 => TensorType::Int32,
            other => TensorType::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    None,
    Dfp,
    Affine,
    Unknown(u32),
}

impl From<u32> for QuantType {
    fn from(v: u32) -> Self {
        match v {
            0 => QuantType::None,
            1 => QuantType::Dfp,
            2 => QuantType::Affine,
            other => QuantType::Unknown(other),
        }
    }
}

impl From<&crate::ffi::RknnTensorAttr> for TensorAttr {
    fn from(raw: &crate::ffi::RknnTensorAttr) -> Self {
        Self {
            index: raw.index,
            shape: raw.shape().to_vec(),
            n_elems: raw.n_elems,
            size: raw.size,
            size_with_stride: raw.size_with_stride,
            format: TensorFormat::from(raw.fmt),
            data_type: TensorType::from(raw.type_),
            qnt_type: QuantType::from(raw.qnt_type),
            zp: raw.zp,
            scale: raw.scale,
            name: raw.name_str().to_string(),
        }
    }
}
