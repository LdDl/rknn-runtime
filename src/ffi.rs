use std::ffi::c_void;

pub type RknnContext = u32;

// Query commands
pub const RKNN_QUERY_IN_OUT_NUM: u32 = 0;
pub const RKNN_QUERY_INPUT_ATTR: u32 = 1;
pub const RKNN_QUERY_OUTPUT_ATTR: u32 = 2;
pub const RKNN_QUERY_SDK_VERSION: u32 = 5;
pub const RKNN_QUERY_NATIVE_INPUT_ATTR: u32 = 8;
pub const RKNN_QUERY_NATIVE_OUTPUT_ATTR: u32 = 9;
pub const RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR: u32 = 10;
pub const RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR: u32 = 11;

// Tensor format constants
pub const RKNN_TENSOR_FORMAT_NCHW: u32 = 0;
pub const RKNN_TENSOR_FORMAT_NHWC: u32 = 1;
pub const RKNN_TENSOR_FORMAT_NC1HWC2: u32 = 2;
pub const RKNN_TENSOR_FORMAT_UNDEFINED: u32 = 0xFFFF;

// Tensor type constants
pub const RKNN_TENSOR_FLOAT32: u32 = 0;
pub const RKNN_TENSOR_FLOAT16: u32 = 1;
pub const RKNN_TENSOR_INT8: u32 = 2;
pub const RKNN_TENSOR_UINT8: u32 = 3;
pub const RKNN_TENSOR_INT16: u32 = 4;
pub const RKNN_TENSOR_INT32: u32 = 5;

// Quantization type constants
pub const RKNN_QUANT_NONE: u32 = 0;
pub const RKNN_QUANT_DFP: u32 = 1;
pub const RKNN_QUANT_AFFINE: u32 = 2;

// Memory sync flags
pub const RKNN_MEM_SYNC_TO_DEVICE: i32 = 0;
pub const RKNN_MEM_SYNC_FROM_DEVICE: i32 = 1;

pub const RKNN_MAX_DIMS: usize = 16;
pub const RKNN_MAX_NAME_LEN: usize = 256;

// For rknn_inputs_set format/type
pub const RKNN_TENSOR_UINT8_INPUT: u8 = 2;
pub const RKNN_TENSOR_NHWC_INPUT: u8 = 0;

#[repr(C)]
pub struct RknnInputOutputNum {
    pub n_input: u32,
    pub n_output: u32,
}

#[repr(C)]
pub struct RknnSdkVersion {
    pub api_version: [u8; 256],
    pub drv_version: [u8; 256],
}

#[repr(C)]
#[derive(Clone)]
pub struct RknnTensorAttr {
    pub index: u32,
    pub n_dims: u32,
    pub dims: [u32; RKNN_MAX_DIMS],
    pub name: [u8; RKNN_MAX_NAME_LEN],
    pub n_elems: u32,
    pub size: u32,
    pub fmt: u32,
    pub type_: u32,
    pub qnt_type: u32,
    pub fl: i8,
    pub zp: i32,
    pub scale: f32,
    pub w_stride: u32,
    pub size_with_stride: u32,
    pub pass_through: u8,
    pub h_stride: u32,
}

impl RknnTensorAttr {
    pub fn new() -> Self {
        unsafe { std::mem::zeroed() }
    }

    pub fn name_str(&self) -> &str {
        std::str::from_utf8(&self.name)
            .unwrap_or("")
            .trim_end_matches('\0')
    }

    pub fn shape(&self) -> &[u32] {
        &self.dims[..self.n_dims as usize]
    }
}

impl Default for RknnTensorAttr {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
pub struct RknnTensorMem {
    pub virt_addr: *mut c_void,
    pub phys_addr: u64,
    pub fd: i32,
    pub offset: i32,
    pub size: u32,
    pub flags: u32,
    pub priv_data: *mut c_void,
}

#[repr(C)]
#[derive(Clone)]
pub struct RknnOutput {
    pub want_float: u8,
    pub is_prealloc: u8,
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
}

impl RknnOutput {
    pub fn new(index: u32) -> Self {
        Self {
            want_float: 0,
            is_prealloc: 0,
            index,
            buf: std::ptr::null_mut(),
            size: 0,
        }
    }
}

#[repr(C)]
pub struct RknnInput {
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
    pub pass_through: u8,
    pub type_: u8,
    pub fmt: u8,
}

// Function type signatures for libloading
pub type FnRknnInit = unsafe extern "C" fn(
    *mut RknnContext,
    *const c_void,
    u32,
    u32,
    *const c_void,
) -> i32;

pub type FnRknnQuery =
    unsafe extern "C" fn(RknnContext, u32, *mut c_void, u32) -> i32;

pub type FnRknnRun =
    unsafe extern "C" fn(RknnContext, *const c_void) -> i32;

pub type FnRknnDestroy = unsafe extern "C" fn(RknnContext) -> i32;

pub type FnRknnCreateMem =
    unsafe extern "C" fn(RknnContext, u32) -> *mut RknnTensorMem;

pub type FnRknnDestroyMem =
    unsafe extern "C" fn(RknnContext, *mut RknnTensorMem) -> i32;

pub type FnRknnSetIoMem = unsafe extern "C" fn(
    RknnContext,
    *mut RknnTensorMem,
    *mut RknnTensorAttr,
) -> i32;

pub type FnRknnMemSync =
    unsafe extern "C" fn(RknnContext, *mut RknnTensorMem, i32) -> i32;

pub type FnRknnInputsSet =
    unsafe extern "C" fn(RknnContext, u32, *mut RknnInput) -> i32;

pub type FnRknnOutputsGet = unsafe extern "C" fn(
    RknnContext,
    u32,
    *mut RknnOutput,
    *const c_void,
) -> i32;

pub type FnRknnOutputsRelease =
    unsafe extern "C" fn(RknnContext, u32, *mut RknnOutput) -> i32;
