use crate::context::RknnContext;
use crate::error::Error;
use crate::memory::ZeroCopyMem;
use crate::tensor::{dequantize_affine, TensorAttr};

const DEFAULT_LIB_PATH: &str = "/usr/lib/librknnmrt.so";

/// High-level RKNN model for zero-copy inference.
///
/// Field order matters: Rust drops fields in declaration order.
/// Memory buffers MUST be dropped before the context, otherwise
/// `rknn_destroy_mem` is called on an already-destroyed context → segfault.
pub struct RknnModel {
    output_mems: Vec<ZeroCopyMem>,
    input_mem: ZeroCopyMem,
    rknn: RknnContext,
    input_attr: TensorAttr,
    output_attrs: Vec<TensorAttr>,
}

impl RknnModel {
    /// Load a model from a file path using the default library location.
    pub fn load(model_path: &str) -> Result<Self, Error> {
        Self::load_with_lib(model_path, DEFAULT_LIB_PATH)
    }

    /// Load a model from a file path with a custom library location.
    pub fn load_with_lib(model_path: &str, lib_path: &str) -> Result<Self, Error> {
        let model_data = std::fs::read(model_path)?;
        Self::load_from_bytes(&model_data, lib_path)
    }

    /// Load a model from raw bytes.
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

    /// Get input tensor attributes.
    pub fn input_attr(&self) -> &TensorAttr {
        &self.input_attr
    }

    /// Get output tensor attributes.
    pub fn output_attrs(&self) -> &[TensorAttr] {
        &self.output_attrs
    }

    /// Run inference with the given input data (raw RGB bytes, NHWC format).
    ///
    /// After calling this, use `output_raw()` or `output_f32()` to read results.
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

    /// Get raw INT8 output data for the given output index.
    pub fn output_raw(&self, index: usize) -> Result<&[i8], Error> {
        if index >= self.output_mems.len() {
            return Err(Error::InvalidIndex {
                requested: index,
                available: self.output_mems.len(),
            });
        }
        Ok(self.output_mems[index].as_i8_slice())
    }

    /// Get dequantized f32 output for the given output index.
    ///
    /// Uses affine dequantization: `value = (raw - zp) * scale`
    pub fn output_f32(&self, index: usize) -> Result<Vec<f32>, Error> {
        let raw = self.output_raw(index)?;
        let attr = &self.output_attrs[index];
        Ok(dequantize_affine(raw, attr.zp, attr.scale))
    }
}
