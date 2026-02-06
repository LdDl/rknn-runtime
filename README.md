# rknn-runtime

Rust bindings for [RKNN](https://github.com/airockchip/rknn-toolkit2) NPU inference on Rockchip SoCs (RV1106, RK3588, etc.)
for object detection tasks for YOLO models (not for classic/traditional one like v3,v4 though :sad_face:).

Load a `.rknn` model, feed it an image, get results back. All the tricky C API details (zero-copy memory, cache sync, tensor layouts) are internal. High-level API is safe I believe, internal - not.

**IMPORTANT NOTE**: I've tested this only for RV1106 on Luckfox Pico Ultra W, using the RKNN Toolkit2 runtime.

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
rknn-runtime = "0.1"
```

Run inference in three steps:

```rust
use rknn_runtime::RknnModel;

// Load model
let model = RknnModel::load("model.rknn")?;

// Run inference (input = raw RGB bytes, NHWC format)
model.run(&rgb_data)?;

// Read output
let raw_i8: &[i8] = model.output_raw(0)?;
// dequantized f32
let floats: Vec<f32> = model.output_f32(0)?;
```

That is it

## What this crate provides

| Item | Description |
|---|---|
| `RknnModel` | Load model, run inference, read outputs |
| `TensorAttr` | Tensor metadata (shape, format, quantization params) |
| `nc1hwc2_to_flat()` | Convert RKNN's packed NC1HWC2 layout to flat NCHW |
| `dequantize_affine()` | Convert INT8 output to f32: `(raw - zp) * scale` |

## How inference works

Q: Here is what happens under the hood when you call `RknnModel::load()` and `run()`?
A: roughly, this:

```
load("model.rknn")
  |-- Load librknnmrt.so (dynamically via libloading, or statically linked). Static linking is not tested by me.
  |-- Call rknn_init() with model bytes
  |-- Query input/output tensor attributes
  `-- Allocate zero-copy memory buffers for input and all outputs

run(&rgb_data)
  |-- Copy RGB bytes into input buffer
  |-- Call rknn_run() (NPU executes the model)
  `-- Call rknn_mem_sync() on each output (sync NPU cache -> CPU)

output_raw(0)  -> &[i8]    - Raw INT8 data, zero-copy, no allocation
output_f32(0)  -> Vec<f32> - Dequantized, allocates new Vec
```

## Input format

RKNN expects raw RGB bytes in **NHWC** layout. No normalization, no channel reordering.

If you have an image file, resize it to the model's input size and convert to RGB:

```rust
let input = model.input_attr();
// NHWC: [1, H, W, 3]
let (h, w) = (input.shape[1], input.shape[2]);

let img = image::open("cat.jpg")?;
let resized = img.resize_exact(w, h, image::imageops::FilterType::Nearest);
let rgb_bytes: Vec<u8> = resized.to_rgb8().into_raw();

model.run(&rgb_bytes)?;
```

## Output format: NC1HWC2

Most RKNN models on RV1106 (my specific case) output tensors in **NC1HWC2** format, not standard NCHW. This is an NPU-specific layout where channels are packed into blocks of `c2` (typically 16, I believe?).

Shape: `[1, c1, H, W, c2]` where `c1 * c2 >= total_channels`.

To make this usable, convert it to a flat NCHW array:

```rust
use rknn_runtime::{nc1hwc2_to_flat, dequantize_affine};

let output = &model.output_attrs()[0];
let (c1, h, w, c2) = (
    output.shape[1] as usize,
    output.shape[2] as usize,
    output.shape[3] as usize,
    output.shape[4] as usize,
);

let raw = model.output_raw(0)?;
let flat = nc1hwc2_to_flat(raw, c1, h, w, c2, total_channels);
let data = dequantize_affine(&flat, output.zp, output.scale);
// data is now [total_channels, H * W] in NCHW order
// access: data[channel * num_predictions + prediction]
```

## Features

| Feature | Description | Default |
|---|---|---|
| `dynamic` | Load `librknnmrt.so` at runtime via [libloading](https://crates.io/crates/libloading). You can compile on x86 without the RKNN library present. | yes |
| `static-link` | Link `librknnmrt.so` at compile time. Requires the library to be available during build. | no |

To use static linking:

```toml
[dependencies]
rknn-runtime = { version = "0.1", default-features = false, features = ["static-link"] }
```

## Cross-compilation

This crate is designed to be compiled on x86 and run on ARM. With the default `dynamic` feature, you don't need the RKNN library on your build machine.

For cross-compilation, I recommend using [cross](https://crates.io/crates/cross), which could be installed with `cargo install cross`.

Then build for the target:
```bash
cross build --target armv7-unknown-linux-gnueabihf --release
```

On the target device, make sure `librknnmrt.so` is available at `/usr/lib/librknnmrt.so` (default path), or specify a custom path:

```rust
let model = RknnModel::load_with_lib("model.rknn", "/opt/lib/librknnmrt.so")?;
```

## INT8 quantization notes

RKNN models are typically quantized to INT8. A couple of things to keep in mind:

**Dequantization.** Raw output is `i8`. To get meaningful float values:

```
value = (raw_i8 - zero_point) * scale
```

The `output_f32()` method and `dequantize_affine()` function do this for you.

**Confidence threshold for detection models.** `sigmoid(0) = 0.5` is the "no opinion" baseline. After INT8 quantization and dequantization, this rounds to ~0.502. If you use 0.5 as your confidence threshold, you'll get thousands of garbage detections. Use **0.51** or higher.

## Supported hardware

Tested on:
- **RV1106** (LuckFox Pico Ultra W) with RKNN Toolkit2 runtime

Should work on other Rockchip SoCs supported by RKNN Toolkit2 (RK3588, RK3566, RK3562, etc.), but not yet tested (I don't have hardware for that, lol).

## Example

See [examples/coco_test](examples/coco_test/) for a complete YOLOv8 COCO object detection example that loads a model, runs inference, decodes NC1HWC2 output, and prints detected objects.
