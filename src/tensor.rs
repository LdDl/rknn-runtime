/// Convert raw INT8 values to f32 using affine dequantization.
///
/// RKNN models quantize weights and activations to INT8 during conversion.
/// Each tensor has a zero-point (`zp`) and scale stored in its
/// [`TensorAttr`]. This function reverses the quantization:
///
/// ```text
/// f32_value = (raw_i8 - zero_point) * scale
/// ```
///
/// # Example
///
/// ```
/// use rknn_runtime::dequantize_affine;
///
/// let raw = vec![10i8, 20, 30];
/// let zp = 5;
/// let scale = 0.1;
/// let result = dequantize_affine(&raw, zp, scale);
/// assert_eq!(result, vec![0.5, 1.5, 2.5]);
/// ```
/// 
pub fn dequantize_affine(data: &[i8], zp: i32, scale: f32) -> Vec<f32> {
    data.iter()
        .map(|&v| (v as f32 - zp as f32) * scale)
        .collect()
}

/// Convert NC1HWC2 tensor layout to flat NCHW order.
///
/// RKNN NPU stores output tensors in a packed format called NC1HWC2.
/// Instead of laying out channels sequentially (like NCHW), it groups
/// them into blocks of `c2` (typically 16):
///
/// ```text
/// NC1HWC2 shape: [1, c1, H, W, c2]
///
/// c1 = ceil(total_channels / c2)
/// Actual channels used: total_channels (the rest is padding)
/// ```
///
/// This function unpacks that into a flat `[total_channels * H * W]` array
/// in standard NCHW order, so you can index it as:
///
/// ```text
/// value = output[channel * H * W + y * W + x]
/// ```
///
/// Works with both `i8` (raw INT8 output) and `f32` (after dequantization).
///
/// # Arguments
///
/// - `data` - raw tensor data in NC1HWC2 layout
/// - `c1` - number of channel blocks (shape\[1\])
/// - `h` - height dimension (shape\[2\])
/// - `w` - width dimension (shape\[3\])
/// - `c2` - channels per block, typically 16 (shape\[4\])
/// - `total_channels` - actual number of channels (e.g. 84 for YOLOv8 with 80 classes)
///
/// # Example
///
/// ```
/// use rknn_runtime::nc1hwc2_to_flat;
///
/// // 4 channels packed into blocks of 2 (c1=2, c2=2), spatial 1x1
/// let nc1hwc2_data: Vec<i8> = vec![
///     10, 20, // block 0: channels 0, 1
///     30, 40, // block 1: channels 2, 3
/// ];
/// let flat = nc1hwc2_to_flat(&nc1hwc2_data, 2, 1, 1, 2, 4);
/// assert_eq!(flat, vec![10, 20, 30, 40]);
/// ```
/// 
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

/// Metadata for a single tensor (input or output).
///
/// Contains everything you need to interpret the tensor data:
/// shape, memory layout, data type, and quantization parameters.
///
/// # Quantization fields
///
/// - `zp` (zero-point) and `scale` are used for INT8 affine dequantization:
///   `f32_value = (raw_i8 - zp) * scale`
/// - These are set during model conversion and are different for each tensor.
///
/// # Shape
///
/// For NC1HWC2 outputs (common on RV1106), the shape is `[1, c1, H, W, c2]`.
/// For NHWC inputs, the shape is `[1, H, W, C]`.
/// 
#[derive(Debug, Clone)]
pub struct TensorAttr {
    /// Tensor index (0 for first input/output, 1 for second, etc.).
    pub index: u32,
    /// Tensor dimensions. See [`TensorFormat`] for how to interpret them.
    pub shape: Vec<u32>,
    /// Total number of elements in the tensor.
    pub n_elems: u32,
    /// Size in bytes.
    pub size: u32,
    /// Size in bytes including stride padding (used for memory allocation).
    pub size_with_stride: u32,
    /// Memory layout of the tensor data.
    pub format: TensorFormat,
    /// Element data type.
    pub data_type: TensorType,
    /// Quantization method.
    pub qnt_type: QuantType,
    /// Quantization zero-point (for affine dequantization).
    pub zp: i32,
    /// Quantization scale (for affine dequantization).
    pub scale: f32,
    /// Human-readable tensor name from the model.
    pub name: String,
}

/// Memory layout of a tensor.
///
/// Describes how tensor data is arranged in memory.
///
/// - **NCHW** - channels first. Common in PyTorch. Shape: `[batch, channels, height, width]`.
/// - **NHWC** - channels last. Used for RKNN inputs. Shape: `[batch, height, width, channels]`.
/// - **NC1HWC2** - RKNN NPU packed format. Channels are split into blocks.
///   Shape: `[batch, c1, height, width, c2]`. Use [`nc1hwc2_to_flat`] to convert.
/// 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFormat {
    /// Channels first: `[N, C, H, W]`.
    NCHW,
    /// Channels last: `[N, H, W, C]`. Standard for RKNN inputs.
    NHWC,
    /// NPU packed format: `[N, c1, H, W, c2]`. Common for RKNN outputs on RV1106.
    NC1HWC2,
    /// Unknown or unsupported format.
    Undefined,
}

/// Just an alias for TensorFormat
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

/// Element data type of a tensor.
///
/// INT8 quantized models (the most common on RKNN) use [`Int8`](Self::Int8) for outputs
/// and [`Uint8`](Self::Uint8) for inputs.
/// 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    Float32,
    Float16,
    Int8,
    Uint8,
    Int16,
    Int32,
    /// Unrecognized type ID from the RKNN runtime.
    Unknown(u32),
}

/// Just an alias for TensorType
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

/// Quantization method used for a tensor.
///
/// Most RKNN INT8 models use [`Affine`](Self::Affine) quantization,
/// where each value is converted via `f32 = (i8 - zp) * scale`.
/// 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// No quantization (float model).
    None,
    /// Dynamic fixed-point quantization.
    Dfp,
    /// Affine quantization: `value = (raw - zp) * scale`. The most common type.
    Affine,
    /// Unrecognized quantization type ID from the RKNN runtime.
    Unknown(u32),
}

/// Just an alias for QuantType
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

/// Precomputed NC1HWC2 tensor layout for efficient, division-free access.
///
/// NC1HWC2 is the RKNN NPU's native output format where channels are packed
/// into blocks of `c2` (typically 16). This struct precomputes the layout
/// parameters at model load time so the per-frame hot path needs zero
/// division operations.
///
/// # Construction
///
/// ```rust,ignore
/// let layout = Nc1hwc2Layout::from_attr(&model.output_attrs()[0])?;
/// ```
///
/// Or use the convenience method on `RknnModel`:
///
/// ```rust,ignore
/// let layout = model.output_nc1hwc2_layout(0)?;
/// ```
///
/// # Channel access
///
/// Each channel has a fixed offset relative to a prediction's base address.
/// For channels 0..c2-1 they live in block 0 (offsets 0..c2-1).
/// Channel `ch` lives at offset `(ch/c2)*hw_c2 + (ch%c2)`.
///
/// Precompute offsets for a range of channels once at load time, then
/// use them in the hot loop with zero division:
///
/// ```rust,ignore
/// // Precompute at load time (once):
/// let class_offsets = layout.precompute_channel_offsets(4, num_classes);
///
/// // Hot loop (per frame):
/// let raw = model.output_raw(0)?;
/// let stride = layout.prediction_stride();
/// let mut p_off = 0;
/// for _p in 0..layout.num_predictions() {
///     for (c, &off) in class_offsets.iter().enumerate() {
///         let v = raw[off + p_off];
///         // ...
///     }
///     p_off += stride;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Nc1hwc2Layout {
    c2: usize,
    /// Size of one c1 block: H * W * c2.
    hw_c2: usize,
    /// Stride between consecutive predictions: W * c2.
    w_c2: usize,
    /// Number of spatial predictions: H * W.
    num_predictions: usize,
    /// Quantization zero-point.
    zp: i32,
    /// Quantization scale.
    scale: f32,
}

impl Nc1hwc2Layout {
    /// Create a layout from a tensor attribute.
    ///
    /// Validates that the tensor is NC1HWC2 with a 5D shape.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidFormat`] if the tensor is not NC1HWC2
    /// or has an unexpected shape.
    pub fn from_attr(attr: &TensorAttr) -> Result<Self, crate::Error> {
        if attr.format != TensorFormat::NC1HWC2 {
            return Err(crate::Error::InvalidFormat {
                expected: "NC1HWC2",
                got: format!("{:?}", attr.format),
            });
        }
        if attr.shape.len() != 5 {
            return Err(crate::Error::InvalidFormat {
                expected: "5D shape [N, C1, H, W, C2]",
                got: format!("{}D shape {:?}", attr.shape.len(), attr.shape),
            });
        }

        let h = attr.shape[2] as usize;
        let w = attr.shape[3] as usize;
        let c2 = attr.shape[4] as usize;

        Ok(Self {
            c2,
            hw_c2: h * w * c2,
            w_c2: w * c2,
            num_predictions: h * w,
            zp: attr.zp,
            scale: attr.scale,
        })
    }

    /// Compute the raw-data offset for channel `ch` relative to a prediction base.
    ///
    /// Equivalent to `(ch / c2) * hw_c2 + (ch % c2)`.
    #[inline]
    pub fn channel_offset(&self, ch: usize) -> usize {
        (ch / self.c2) * self.hw_c2 + (ch % self.c2)
    }

    /// Precompute offsets for `count` consecutive channels starting from `start_ch`.
    ///
    /// Returns a `Vec` where `result[i]` is the offset for channel `start_ch + i`.
    /// Use this at load time to build a lookup table for the hot loop.
    pub fn precompute_channel_offsets(&self, start_ch: usize, count: usize) -> Vec<usize> {
        (0..count)
            .map(|i| self.channel_offset(start_ch + i))
            .collect()
    }

    /// Stride (in i8 elements) between consecutive spatial predictions.
    ///
    /// In the hot loop, increment `p_offset += stride` instead of
    /// multiplying `p * stride` each iteration.
    #[inline]
    pub fn prediction_stride(&self) -> usize {
        self.w_c2
    }

    /// Number of spatial predictions (H * W).
    #[inline]
    pub fn num_predictions(&self) -> usize {
        self.num_predictions
    }

    /// Channel block size (typically 16).
    #[inline]
    pub fn c2(&self) -> usize {
        self.c2
    }

    /// Quantization zero-point.
    #[inline]
    pub fn zp(&self) -> i32 {
        self.zp
    }

    /// Quantization scale factor.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Dequantize a single raw i8 value to f32.
    ///
    /// `f32_value = (raw_i8 - zp) * scale`
    #[inline]
    pub fn dequant(&self, raw: i8) -> f32 {
        (raw as i32 - self.zp) as f32 * self.scale
    }

    /// Convert a f32 confidence threshold to i8 space for fast rejection.
    ///
    /// When `scale > 0`, the affine mapping is monotonically increasing,
    /// so we can compare raw i8 values directly without dequantization.
    /// This rejects ~99% of predictions with zero float math.
    ///
    /// Returns `i8::MIN` if scale <= 0 (passes everything through).
    #[inline]
    pub fn threshold_i8(&self, conf: f32) -> i8 {
        if self.scale > 0.0 {
            let raw_f = conf / self.scale + self.zp as f32;
            (raw_f.ceil() as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8
        } else {
            i8::MIN
        }
    }
}

/// Implements `From` for TensorAttr
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
