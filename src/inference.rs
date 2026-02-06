//! High-level inference API.
//!
//! This module contains [`RknnModel`] - the main entry point for loading
//! and running RKNN models.

use crate::context::RknnContext;
use crate::error::Error;
use crate::memory::ZeroCopyMem;
use crate::tensor::{dequantize_affine, Nc1hwc2Layout, TensorAttr};

const DEFAULT_LIB_PATH: &str = "/usr/lib/librknnmrt.so";

/// A loaded RKNN model ready for inference.
///
/// This is the main type you interact with. It holds the model, pre-allocated
/// zero-copy memory buffers for input and outputs, and handles all
/// communication with the NPU.
///
/// # Lifecycle
///
/// 1. **Load** a model with [`load`](Self::load) or [`load_with_lib`](Self::load_with_lib).
///    This initializes the NPU context and allocates memory buffers.
/// 2. **Inspect** tensor metadata via [`input_attr`](Self::input_attr) and
///    [`output_attrs`](Self::output_attrs) to learn expected shapes and formats.
/// 3. **Run** inference with [`run`](Self::run) - pass raw RGB bytes (NHWC, u8,
///    no normalization).
/// 4. **Read** results with [`output_raw`](Self::output_raw) (zero-copy `&[i8]`)
///    or [`output_f32`](Self::output_f32) (dequantized `Vec<f32>`).
///
/// # Example
///
/// ```rust,no_run
/// use rknn_runtime::RknnModel;
///
/// let model = RknnModel::load("model.rknn")?;
///
/// // Check what the model expects
/// let input = model.input_attr();
/// // e.g. [1, 320, 320, 3]
/// println!("Input shape: {:?}", input.shape);
///
/// // Run inference
/// # let rgb_bytes = vec![0u8; 320 * 320 * 3];
/// model.run(&rgb_bytes)?;
///
/// // Get raw INT8 output (zero-copy - no allocation, just a slice into NPU memory)
/// let raw = model.output_raw(0)?;
///
/// // Or get dequantized f32 output (allocates a new Vec)
/// let floats = model.output_f32(0)?;
/// # Ok::<(), rknn_runtime::Error>(())
/// ```
///
/// # Drop order
///
/// Internally, memory buffers are dropped before the RKNN context.
/// This is handled automatically - you don't need to worry about it.
pub struct RknnModel {
    output_mems: Vec<ZeroCopyMem>,
    input_mem: ZeroCopyMem,
    rknn: RknnContext,
    input_attr: TensorAttr,
    output_attrs: Vec<TensorAttr>,
}

impl RknnModel {
    /// Load a `.rknn` model from a file.
    ///
    /// Uses the default library path (`/usr/lib/librknnmrt.so`).
    /// If your `librknnmrt.so` is elsewhere, use [`load_with_lib`](Self::load_with_lib).
    ///
    /// # Errors
    ///
    /// - [`Error::IoError`] if the file cannot be read.
    /// - [`Error::LibraryNotFound`] if `librknnmrt.so` is not found.
    /// - [`Error::InitFailed`] if the NPU rejects the model.
    pub fn load(model_path: &str) -> Result<Self, Error> {
        Self::load_with_lib(model_path, DEFAULT_LIB_PATH)
    }

    /// Load a `.rknn` model from a file, using a custom library path.
    ///
    /// ```rust,no_run
    /// # use rknn_runtime::RknnModel;
    /// let model = RknnModel::load_with_lib(
    ///     "model.rknn",
    ///     "/opt/rknn/lib/librknnmrt.so",
    /// )?;
    /// # Ok::<(), rknn_runtime::Error>(())
    /// ```
    pub fn load_with_lib(model_path: &str, lib_path: &str) -> Result<Self, Error> {
        let model_data = std::fs::read(model_path)?;
        Self::load_from_bytes(&model_data, lib_path)
    }

    /// Load a model from raw bytes already in memory.
    ///
    /// Useful when the `.rknn` file is embedded in your binary or received
    /// over the network.
    pub fn load_from_bytes(model_data: &[u8], lib_path: &str) -> Result<Self, Error> {
        let rknn = RknnContext::load(model_data, lib_path)?;
        let (n_input, n_output) = rknn.query_io_num()?;

        // Query input attributes (NHWC native for zero-copy)
        // We only support single-input models for now
        let raw_input_attr = rknn.query_input_attr_nhwc(0)?;
        let input_attr = TensorAttr::from(&raw_input_attr);

        // Query output attributes (native format for zero-copy)
        let mut raw_output_attrs = Vec::with_capacity(n_output as usize);
        let mut output_attrs = Vec::with_capacity(n_output as usize);
        for i in 0..n_output {
            let attr = rknn.query_output_attr_native(i)?;
            output_attrs.push(TensorAttr::from(&attr));
            raw_output_attrs.push(attr);
        }

        // Allocate zero-copy memory
        let input_mem = ZeroCopyMem::new(&rknn, raw_input_attr)?;

        let mut output_mems = Vec::with_capacity(n_output as usize);
        for attr in raw_output_attrs {
            output_mems.push(ZeroCopyMem::new(&rknn, attr)?);
        }

        // Suppress unused variable warning for single-input models
        let _ = n_input;

        Ok(Self {
            rknn,
            input_mem,
            output_mems,
            input_attr,
            output_attrs,
        })
    }

    /// Input tensor metadata (shape, format, data type).
    ///
    /// The shape is typically `[1, H, W, 3]` (NHWC).
    /// Use this to know what image size the model expects:
    ///
    /// ```rust,no_run
    /// # use rknn_runtime::RknnModel;
    /// # let model = RknnModel::load("m.rknn").unwrap();
    /// let input = model.input_attr();
    /// let (h, w) = (input.shape[1], input.shape[2]);
    /// println!("Model expects {}x{} RGB image", h, w);
    /// ```
    pub fn input_attr(&self) -> &TensorAttr {
        &self.input_attr
    }

    /// Output tensor metadata for all outputs.
    ///
    /// Most models have a single output, but some could have several.
    /// Each [`TensorAttr`] contains the shape, format, quantization zero-point
    /// and scale - everything you need to decode the output.
    pub fn output_attrs(&self) -> &[TensorAttr] {
        &self.output_attrs
    }

    /// Run inference on the NPU.
    ///
    /// `input` must be raw RGB bytes in **NHWC** format (`[1, H, W, 3]`).
    /// No normalization, no channel reordering - just plain `u8` pixel values.
    ///
    /// After this returns, read results with [`output_raw`](Self::output_raw)
    /// or [`output_f32`](Self::output_f32).
    ///
    /// # What happens inside
    ///
    /// 1. Copies `input` bytes into the pre-allocated NPU input buffer.
    /// 2. Calls `rknn_run()` - the NPU executes the model.
    /// 3. Calls `rknn_mem_sync()` on each output buffer (syncs NPU cache to CPU).
    /// In my case: this step is critical on RV1106 - without it, I get stale data.
    ///
    pub fn run(&self, input: &[u8]) -> Result<(), Error> {
        // Copy input data to zero-copy memory
        self.input_mem.write(input);

        // Run NPU inference
        let ret = unsafe {
            (self.rknn.funcs.run)(self.rknn.ctx, std::ptr::null())
        };
        if ret != 0 {
            return Err(Error::InferenceFailed(ret));
        }

        // Sync all output memories from NPU to CPU
        for mem in &self.output_mems {
            mem.sync_from_device(&self.rknn)?;
        }

        Ok(())
    }

    /// Raw INT8 output data for the given output index.
    ///
    /// Returns a slice pointing directly into the NPU's zero-copy buffer.
    /// No allocation, no copying - this is as fast as it gets.
    ///
    /// The data is in whatever layout the NPU uses (often NC1HWC2).
    /// Use [`nc1hwc2_to_flat`](crate::nc1hwc2_to_flat) to convert it
    /// to standard NCHW if needed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidIndex`] if `index` is out of range.
    /// 
    pub fn output_raw(&self, index: usize) -> Result<&[i8], Error> {
        if index >= self.output_mems.len() {
            return Err(Error::InvalidIndex {
                requested: index,
                available: self.output_mems.len(),
            });
        }
        Ok(self.output_mems[index].as_i8_slice())
    }

    /// Precomputed NC1HWC2 layout for the given output index.
    ///
    /// Returns an [`Nc1hwc2Layout`] with all shape and quantization parameters
    /// precomputed. Use this at model load time to prepare channel offset tables,
    /// then use them in the per-image (frame of video, most of time) hot loop with zero division.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidIndex`] if `index` is out of range.
    /// - [`Error::InvalidFormat`] if the output is not NC1HWC2.
    pub fn output_nc1hwc2_layout(&self, index: usize) -> Result<Nc1hwc2Layout, Error> {
        if index >= self.output_attrs.len() {
            return Err(Error::InvalidIndex {
                requested: index,
                available: self.output_attrs.len(),
            });
        }
        Nc1hwc2Layout::from_attr(&self.output_attrs[index])
    }

    /// Dequantized f32 output for the given output index.
    ///
    /// Converts each raw INT8 value to f32 using affine dequantization:
    ///
    /// ```text
    /// value = (raw_i8 - zero_point) * scale
    /// ```
    ///
    /// Zero-point and scale are read from the tensor's quantization parameters
    /// (set during model conversion).
    ///
    /// **Note:** This allocates a new `Vec<f32>`. If you need to dequantize
    /// only part of the output (e.g. after NC1HWC2 conversion), use
    /// [`dequantize_affine`] directly.
    /// 
    pub fn output_f32(&self, index: usize) -> Result<Vec<f32>, Error> {
        let raw = self.output_raw(index)?;
        let attr = &self.output_attrs[index];
        Ok(dequantize_affine(raw, attr.zp, attr.scale))
    }
}
