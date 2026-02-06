//! YOLOv8 COCO inference example for RKNN NPU.
//!
//! Usage: cargo run --example coco_test -- <model.rknn> <image.jpg> [conf_threshold]

use rknn_runtime::{RknnModel, TensorFormat, nc1hwc2_to_flat, dequantize_affine};
use std::time::Instant;

const COCO_NAMES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// cx, cy, w, h, conf, class_id
type Detection = (f32, f32, f32, f32, f32, usize);

/// Computes intersection over Union (IoU) of two detections
fn compute_iou(a: &Detection, b: &Detection) -> f32 {
    let (ax1, ay1, ax2, ay2) = (a.0 - a.2 / 2.0, a.1 - a.3 / 2.0, a.0 + a.2 / 2.0, a.1 + a.3 / 2.0);
    let (bx1, by1, bx2, by2) = (b.0 - b.2 / 2.0, b.1 - b.3 / 2.0, b.0 + b.2 / 2.0, b.1 + b.3 / 2.0);

    let inter_w = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let inter_h = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = inter_w * inter_h;
    let union = a.2 * a.3 + b.2 * b.3 - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Non-Maximum Suppression (NMS) to filter overlapping detections. It is pretty naive implementation which is O(n^2).
/// But should work for demo
fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    detections.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    let mut i = 0;
    while i < detections.len() {
        let mut j = i + 1;
        while j < detections.len() {
            if compute_iou(&detections[i], &detections[j]) > iou_threshold {
                detections.swap_remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: coco_test <model.rknn> <image.jpg> [conf_threshold]");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let image_path = &args[2];
    // Default 0.51: sigmoid(0)=0.5 is the "no opinion" baseline, which INT8
    // dequantization rounds to ~0.502. Threshold must be above that.
    let conf_threshold: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.51);

    // Load model
    let model = RknnModel::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });

    let input = model.input_attr();
    let output = &model.output_attrs()[0];
    let (in_h, in_w) = (input.shape[1], input.shape[2]);

    println!("Model: {}", model_path);
    println!("Input: {}x{} NHWC", in_h, in_w);
    println!("Output: {:?} {:?}", output.shape, output.format);

    // Validate: expect single YOLOv8 NC1HWC2 output
    if model.output_attrs().len() != 1 || output.format != TensorFormat::NC1HWC2 {
        eprintln!("Expected single NC1HWC2 output (YOLOv8 format)");
        std::process::exit(1);
    }

    // Load and resize image
    let img = image::open(image_path).expect("Failed to open image");
    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
    let resized = img.resize_exact(in_w, in_h, image::imageops::FilterType::Nearest);
    let input_data: Vec<u8> = resized.to_rgb8().into_raw();

    // Warmup + benchmark
    for _ in 0..3 { model.run(&input_data).unwrap(); }

    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations { model.run(&input_data).unwrap(); }
    let avg_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    println!("{:.1} ms/inference ({:.1} FPS)", avg_ms, 1000.0 / avg_ms);

    // Convert NC1HWC2 → flat NCHW, then dequantize
    let num_classes = COCO_NAMES.len();
    let total_channels = 4 + num_classes; // 84
    let (c1, h_dim, w_dim, c2) = (
        output.shape[1] as usize,
        output.shape[2] as usize,
        output.shape[3] as usize,
        output.shape[4] as usize,
    );
    let num_predictions = h_dim; // H dimension = number of predictions

    let raw = model.output_raw(0).unwrap();
    let flat = nc1hwc2_to_flat(raw, c1, h_dim, w_dim, c2, total_channels);
    let data = dequantize_affine(&flat, output.zp, output.scale);
    // data is now [total_channels, num_predictions] in NCHW order
    // access: data[ch * num_predictions + p]

    println!("{} predictions, {} classes", num_predictions, num_classes);

    // Decode YOLOv8 output (bbox normalized 0-1, class scores are sigmoid probs)
    // Scale directly to original image coordinates
    let mut detections = Vec::new();

    for p in 0..num_predictions {
        let (mut best_cls, mut best_conf) = (0, 0.0f32);
        for c in 0..num_classes {
            let prob = data[(4 + c) * num_predictions + p];
            if prob > best_conf {
                best_conf = prob;
                best_cls = c;
            }
        }
        if best_conf < conf_threshold { continue; }

        // bbox is normalized 0-1, scale to original image size
        let cx = data[0 * num_predictions + p] * orig_w;
        let cy = data[1 * num_predictions + p] * orig_h;
        let w = data[2 * num_predictions + p] * orig_w;
        let h = data[3 * num_predictions + p] * orig_h;

        if w > 0.0 && h > 0.0 {
            detections.push((cx, cy, w, h, best_conf, best_cls));
        }
    }

    let before = detections.len();
    nms(&mut detections, 0.45);
    println!("{} -> {} detections (after NMS)\n", before, detections.len());

    for (i, &(cx, cy, w, h, conf, cls)) in detections.iter().enumerate() {
        let name = COCO_NAMES.get(cls).unwrap_or(&"?");
        println!("  {:2}. {} ({:.1}%) at ({:.0}, {:.0}) {:.0}x{:.0}",
                 i + 1, name, conf * 100.0, cx, cy, w, h);
    }
}
