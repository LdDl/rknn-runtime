// Low-level RKNN context: library loading, model init, attribute queries.
// This is internal - users interact with `RknnModel` in inference.rs.

use std::ffi::c_void;

use crate::error::Error;
use crate::ffi::*;

/// Function pointers resolved from librknnmrt.so (or linked statically).
pub(crate) struct RknnFunctions {
    pub init: FnRknnInit,
    pub query: FnRknnQuery,
    pub run: FnRknnRun,
    pub destroy: FnRknnDestroy,
    pub create_mem: FnRknnCreateMem,
    pub destroy_mem: FnRknnDestroyMem,
    pub set_io_mem: FnRknnSetIoMem,
    pub mem_sync: FnRknnMemSync,
}

/// RKNN context: holds the NPU handle and resolved function pointers.
///
/// Created during `RknnModel::load()`. Destroyed on drop (calls `rknn_destroy`).
/// The `_lib` field keeps the dynamically loaded library alive.
pub(crate) struct RknnContext {
    pub ctx: crate::ffi::RknnContext,
    pub funcs: RknnFunctions,
    #[cfg(feature = "dynamic")]
    pub _lib: libloading::Library,
}

impl RknnContext {
    /// Load librknnmrt.so at runtime and init the model.
    #[cfg(feature = "dynamic")]
    pub fn load(model_data: &[u8], lib_path: &str) -> Result<Self, Error> {
        if model_data.is_empty() {
            return Err(Error::InvalidModel);
        }

        let lib = unsafe {
            libloading::Library::new(lib_path)
                .map_err(|_| Error::LibraryNotFound(lib_path.to_string()))?
        };

        let funcs = unsafe {
            RknnFunctions {
                init: *lib
                    .get::<FnRknnInit>(b"rknn_init")
                    .map_err(|_| Error::SymbolNotFound("rknn_init".into()))?,
                query: *lib
                    .get::<FnRknnQuery>(b"rknn_query")
                    .map_err(|_| Error::SymbolNotFound("rknn_query".into()))?,
                run: *lib
                    .get::<FnRknnRun>(b"rknn_run")
                    .map_err(|_| Error::SymbolNotFound("rknn_run".into()))?,
                destroy: *lib
                    .get::<FnRknnDestroy>(b"rknn_destroy")
                    .map_err(|_| Error::SymbolNotFound("rknn_destroy".into()))?,
                create_mem: *lib
                    .get::<FnRknnCreateMem>(b"rknn_create_mem")
                    .map_err(|_| Error::SymbolNotFound("rknn_create_mem".into()))?,
                destroy_mem: *lib
                    .get::<FnRknnDestroyMem>(b"rknn_destroy_mem")
                    .map_err(|_| Error::SymbolNotFound("rknn_destroy_mem".into()))?,
                set_io_mem: *lib
                    .get::<FnRknnSetIoMem>(b"rknn_set_io_mem")
                    .map_err(|_| Error::SymbolNotFound("rknn_set_io_mem".into()))?,
                mem_sync: *lib
                    .get::<FnRknnMemSync>(b"rknn_mem_sync")
                    .map_err(|_| Error::SymbolNotFound("rknn_mem_sync".into()))?,
            }
        };

        let mut ctx: crate::ffi::RknnContext = 0;
        let ret = unsafe {
            (funcs.init)(
                &mut ctx,
                model_data.as_ptr() as *const c_void,
                model_data.len() as u32,
                0,
                std::ptr::null(),
            )
        };
        if ret != 0 {
            return Err(Error::InitFailed(ret));
        }

        Ok(Self {
            ctx,
            funcs,
            _lib: lib,
        })
    }

    /// Static-link variant: symbols resolved at compile time.
    #[cfg(feature = "static-link")]
    pub fn load(model_data: &[u8], _lib_path: &str) -> Result<Self, Error> {
        if model_data.is_empty() {
            return Err(Error::InvalidModel);
        }

        extern "C" {
            fn rknn_init(
                ctx: *mut crate::ffi::RknnContext,
                model: *const c_void,
                size: u32,
                flag: u32,
                extend: *const c_void,
            ) -> i32;
            fn rknn_query(
                ctx: crate::ffi::RknnContext,
                cmd: u32,
                info: *mut c_void,
                size: u32,
            ) -> i32;
            fn rknn_run(
                ctx: crate::ffi::RknnContext,
                extend: *const c_void,
            ) -> i32;
            fn rknn_destroy(ctx: crate::ffi::RknnContext) -> i32;
            fn rknn_create_mem(
                ctx: crate::ffi::RknnContext,
                size: u32,
            ) -> *mut RknnTensorMem;
            fn rknn_destroy_mem(
                ctx: crate::ffi::RknnContext,
                mem: *mut RknnTensorMem,
            ) -> i32;
            fn rknn_set_io_mem(
                ctx: crate::ffi::RknnContext,
                mem: *mut RknnTensorMem,
                attr: *mut RknnTensorAttr,
            ) -> i32;
            fn rknn_mem_sync(
                ctx: crate::ffi::RknnContext,
                mem: *mut RknnTensorMem,
                flags: i32,
            ) -> i32;
        }

        let funcs = RknnFunctions {
            init: rknn_init,
            query: rknn_query,
            run: rknn_run,
            destroy: rknn_destroy,
            create_mem: rknn_create_mem,
            destroy_mem: rknn_destroy_mem,
            set_io_mem: rknn_set_io_mem,
            mem_sync: rknn_mem_sync,
        };

        let mut ctx: crate::ffi::RknnContext = 0;
        let ret = unsafe {
            (funcs.init)(
                &mut ctx,
                model_data.as_ptr() as *const c_void,
                model_data.len() as u32,
                0,
                std::ptr::null(),
            )
        };
        if ret != 0 {
            return Err(Error::InitFailed(ret));
        }

        Ok(Self { ctx, funcs })
    }

    /// Query how many inputs and outputs the model has.
    pub fn query_io_num(&self) -> Result<(u32, u32), Error> {
        let mut io_num = RknnInputOutputNum {
            n_input: 0,
            n_output: 0,
        };
        let ret = unsafe {
            (self.funcs.query)(
                self.ctx,
                RKNN_QUERY_IN_OUT_NUM,
                &mut io_num as *mut _ as *mut c_void,
                std::mem::size_of::<RknnInputOutputNum>() as u32,
            )
        };
        if ret != 0 {
            return Err(Error::QueryFailed(ret));
        }
        Ok((io_num.n_input, io_num.n_output))
    }

    /// Query input tensor attributes in NHWC layout (native for zero-copy input).
    pub fn query_input_attr_nhwc(&self, index: u32) -> Result<RknnTensorAttr, Error> {
        let mut attr = RknnTensorAttr::new();
        attr.index = index;
        let ret = unsafe {
            (self.funcs.query)(
                self.ctx,
                RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,
                &mut attr as *mut _ as *mut c_void,
                std::mem::size_of::<RknnTensorAttr>() as u32,
            )
        };
        if ret != 0 {
            return Err(Error::QueryFailed(ret));
        }
        Ok(attr)
    }

    /// Query output tensor attributes in NPU-native layout (NC1HWC2 on RV1106).
    pub fn query_output_attr_native(&self, index: u32) -> Result<RknnTensorAttr, Error> {
        let mut attr = RknnTensorAttr::new();
        attr.index = index;
        let ret = unsafe {
            (self.funcs.query)(
                self.ctx,
                RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                &mut attr as *mut _ as *mut c_void,
                std::mem::size_of::<RknnTensorAttr>() as u32,
            )
        };
        if ret != 0 {
            return Err(Error::QueryFailed(ret));
        }
        Ok(attr)
    }
}

impl Drop for RknnContext {
    fn drop(&mut self) {
        unsafe {
            (self.funcs.destroy)(self.ctx);
        }
    }
}
