use std::fmt;

#[derive(Debug)]
pub enum Error {
    LibraryNotFound(String),
    SymbolNotFound(String),
    InitFailed(i32),
    QueryFailed(i32),
    InferenceFailed(i32),
    MemAllocFailed,
    MemSyncFailed(i32),
    SetIoMemFailed(i32),
    InvalidModel,
    InvalidIndex { requested: usize, available: usize },
    IoError(std::io::Error),
}

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

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IoError(e)
    }
}
