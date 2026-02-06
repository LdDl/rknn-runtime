use crate::context::RknnContext;
use crate::error::Error;
use crate::ffi::*;

/// A zero-copy memory buffer allocated by the RKNN runtime.
/// Automatically freed on drop.
pub(crate) struct ZeroCopyMem {
    pub mem: *mut RknnTensorMem,
    pub attr: RknnTensorAttr,
    ctx_handle: crate::ffi::RknnContext,
    destroy_fn: FnRknnDestroyMem,
}

impl ZeroCopyMem {
    /// Allocate a zero-copy memory buffer and bind it to the given tensor.
    pub fn new(rknn: &RknnContext, mut attr: RknnTensorAttr) -> Result<Self, Error> {
        let alloc_size = attr.size_with_stride.max(attr.n_elems);
        let mem = unsafe { (rknn.funcs.create_mem)(rknn.ctx, alloc_size) };
        if mem.is_null() {
            return Err(Error::MemAllocFailed);
        }

        let ret = unsafe { (rknn.funcs.set_io_mem)(rknn.ctx, mem, &mut attr) };
        if ret != 0 {
            unsafe { (rknn.funcs.destroy_mem)(rknn.ctx, mem) };
            return Err(Error::SetIoMemFailed(ret));
        }

        Ok(Self {
            mem,
            attr,
            ctx_handle: rknn.ctx,
            destroy_fn: rknn.funcs.destroy_mem,
        })
    }

    /// Write data into the zero-copy buffer.
    pub fn write(&self, data: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (*self.mem).virt_addr as *mut u8,
                data.len().min((*self.mem).size as usize),
            );
        }
    }

    /// Sync memory from NPU to CPU (read output after inference).
    pub fn sync_from_device(&self, rknn: &RknnContext) -> Result<(), Error> {
        let ret = unsafe {
            (rknn.funcs.mem_sync)(rknn.ctx, self.mem, RKNN_MEM_SYNC_FROM_DEVICE)
        };
        if ret != 0 {
            return Err(Error::MemSyncFailed(ret));
        }
        Ok(())
    }

    /// Get a raw byte slice of the output data.
    pub fn as_i8_slice(&self) -> &[i8] {
        unsafe {
            std::slice::from_raw_parts(
                (*self.mem).virt_addr as *const i8,
                self.attr.size_with_stride as usize,
            )
        }
    }
}

impl Drop for ZeroCopyMem {
    fn drop(&mut self) {
        unsafe {
            (self.destroy_fn)(self.ctx_handle, self.mem);
        }
    }
}
