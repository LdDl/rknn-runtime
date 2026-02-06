Compile:
```
cross build --target armv7-unknown-linux-gnueabihf --release --example coco_test
```

Run on target device:
```bash
## Depending on your setup, you might need to use `sudo` to run the executable, especially if it needs access to certain hardware resources
# ./coco_test yolov8n.rknn cat.jpg 0.6
sudo ./coco_test yolov8n.rknn cat.jpg 0.6
```

If everything fine you will see something like this:
```shell
Model: yolov8n.rknn
Input: 320x320 NHWC
Output: [1, 6, 2100, 1, 16] NC1HWC2
37.8 ms/inference (26.5 FPS)
2100 predictions, 80 classes
2100 -> 463 detections (after NMS)

   1. cat (68.0%) at (338, 199) 279x167
   2. car (62.9%) at (328, 553) 627x553
   3. cat (65.5%) at (309, 254) 240x154
```

Output bbox coordinates are in format (cx, cy, w, h) where cx and cy are center of the box. In this example they are already scaled for input image size.

In this example I've got `yolov8n.rknn` following this instructions [rv1106-yolov8](https://github.com/LdDl/rv1106-yolov8).

Note that accuracy of the model is dependent on the calibration dataset, which is important due FP32 -> F8 quantization. So if you want to get better accuracy, you should use more representative calibration dataset. In this example I've just used 128 image from COCO validation set.

