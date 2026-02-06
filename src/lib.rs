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
    dequantize_affine, nc1hwc2_to_flat,
};
