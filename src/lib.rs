//! Rust bindings for RKNN NPU inference on Rockchip SoCs.
//!
//! This crate allows you to load `*.rknn` model, run it on the NPU, and read back
//! the results. It wraps the C library `librknnmrt.so` and handles all the
//! low-level details: zero-copy memory allocation, NPU-to-CPU cache sync,
//! and the unusual NC1HWC2 tensor layout.
//!
//! # Usage
//!
//! ```rust,no_run
//! use rknn_runtime::RknnModel;
//!
//! // Load a model file
//! let model = RknnModel::load("model.rknn").unwrap();
//!
//! // Prepare input: raw RGB bytes in NHWC layout, no normalization needed.
//! // The byte length must match the model's expected input size.
//! # let rgb_data: Vec<u8> = vec![0u8; 320 * 320 * 3];
//!
//! // Run inference on the NPU
//! model.run(&rgb_data).unwrap();
//!
//! // Read output as raw INT8 (zero-copy, no allocation)
//! let raw: &[i8] = model.output_raw(0).unwrap();
//!
//! // ...or as dequantized f32 (allocates a new Vec)
//! let floats: Vec<f32> = model.output_f32(0).unwrap();
//! ```
//!
//! # NC1HWC2 output layout
//!
//! RKNN models (especially on RV1106) often output tensors in NC1HWC2 format
//! instead of standard NCHW. Channels are packed into blocks of `c2`
//! (typically 16). Use [`nc1hwc2_to_flat`] to convert this into a normal
//! flat array before parsing:
//!
//! ```rust,ignore
//! let flat = nc1hwc2_to_flat(raw, c1, h, w, c2, total_channels);
//! let data = dequantize_affine(&flat, output.zp, output.scale);
//! // data[channel * num_predictions + prediction_index]
//! ```
//!
//! # Linking modes
//!
//! - **`dynamic`** (default) - loads `librknnmrt.so` at runtime via
//!   [`libloading`](https://crates.io/crates/libloading). You can compile
//!   on x86 without having the RKNN library installed.
//! - **`static-link`** - links at compile time. Requires `librknnmrt.so`
//!   to be present on the build machine.

pub mod ffi;
pub mod error;
mod context;
mod memory;
pub mod inference;
pub mod tensor;

pub use error::Error;
pub use inference::RknnModel;
pub use tensor::{
    TensorAttr, TensorFormat, TensorType, QuantType,
    Nc1hwc2Layout,
    dequantize_affine, nc1hwc2_to_flat,
};
