//! Error types for RKNN operations.

use std::fmt;

/// Everything that can go wrong when using RKNN.
///
/// Most variants carry the RKNN error code (a negative `i32`).
/// The RKNN SDK doesn't document specific codes, but common ones are:
/// - `-1` - generic failure
/// - `-2` - invalid parameter
/// - `-3` - device not found
///
/// # Example
///
/// ```rust,no_run
/// use rknn_runtime::{RknnModel, Error};
///
/// match RknnModel::load("model.rknn") {
///     Ok(model) => println!("Loaded!"),
///     Err(Error::LibraryNotFound(path)) => {
///         eprintln!("RKNN library not found at: {}", path);
///         eprintln!("Make sure librknnmrt.so is installed on the device.");
///     }
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
#[derive(Debug)]
pub enum Error {
    /// `librknnmrt.so` could not be loaded. Contains the attempted path.
    LibraryNotFound(String),
    /// A required function was not found in the loaded library.
    SymbolNotFound(String),
    /// `rknn_init()` failed. The model file may be corrupt or incompatible.
    InitFailed(i32),
    /// `rknn_query()` failed when reading tensor metadata.
    QueryFailed(i32),
    /// `rknn_run()` failed during inference.
    InferenceFailed(i32),
    /// NPU memory allocation returned null.
    MemAllocFailed,
    /// `rknn_mem_sync()` failed. Cache sync between NPU and CPU did not complete.
    MemSyncFailed(i32),
    /// `rknn_set_io_mem()` failed when binding a memory buffer to a tensor.
    SetIoMemFailed(i32),
    /// The model data is empty or invalid.
    InvalidModel,
    /// Output index is out of range.
    InvalidIndex {
        /// The index that was requested.
        requested: usize,
        /// How many outputs the model actually has.
        available: usize,
    },
    /// Tensor format or shape does not match what was expected.
    InvalidFormat {
        /// What was expected (e.g. "NC1HWC2").
        expected: &'static str,
        /// What was actually found.
        got: String,
    },
    /// File I/O error (e.g. model file not found).
    IoError(std::io::Error),
}

/// Just display the error message.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LibraryNotFound(path) => {
                write!(f, "failed to load library: {}", path)
            }
            Error::SymbolNotFound(name) => {
                write!(f, "symbol not found: {}", name)
            }
            Error::InitFailed(code) => {
                write!(f, "rknn_init failed (code {})", code)
            }
            Error::QueryFailed(code) => {
                write!(f, "rknn_query failed (code {})", code)
            }
            Error::InferenceFailed(code) => {
                write!(f, "rknn_run failed (code {})", code)
            }
            Error::MemAllocFailed => {
                write!(f, "rknn_create_mem returned null")
            }
            Error::MemSyncFailed(code) => {
                write!(f, "rknn_mem_sync failed (code {})", code)
            }
            Error::SetIoMemFailed(code) => {
                write!(f, "rknn_set_io_mem failed (code {})", code)
            }
            Error::InvalidFormat { expected, got } => {
                write!(f, "expected {} format, got {}", expected, got)
            }
            Error::InvalidModel => write!(f, "invalid or empty model data"),
            Error::InvalidIndex {
                requested,
                available,
            } => {
                write!(
                    f,
                    "output index {} out of range (have {})",
                    requested, available
                )
            }
            Error::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

/// Allow chaining
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IoError(e) => Some(e),
            _ => None,
        }
    }
}

/// Convert std::io::Error into crate's Error type.
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IoError(e)
    }
}
