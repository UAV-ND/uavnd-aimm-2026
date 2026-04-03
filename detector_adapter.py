#!/usr/bin/env python3
import time
import cv2
import Jetson.GPIO as GPIO

from cv_2 import TensorRTInfer, preprocess, postprocess, gstreamer_pipeline

PAYLOAD_PIN = 16
PAYLOAD_PULSE_S = 0.5


class GenericDetector:
    def __init__(
        self,
        engine_path,
        smooth_alpha=0.12,
        trigger_conf=0.97,
        stable_dist_px=100,
        area_ratio_min=0.55,
        area_ratio_max=1.80,
        enable_payload_gpio=False
    ):
        self.engine_path = engine_path
        self.smooth_alpha = smooth_alpha
        self.trigger_conf = trigger_conf
        self.stable_dist_px = stable_dist_px
        self.area_ratio_min = area_ratio_min
        self.area_ratio_max = area_ratio_max
        self.enable_payload_gpio = enable_payload_gpio

        print("Loading TensorRT engine:", self.engine_path)
        self.trt_infer = TensorRTInfer(self.engine_path)

        pipeline = gstreamer_pipeline(flip_method=0)
        print("Using pipeline:")
        print(pipeline)

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera")

        self.prev_center = None
        self.prev_area = None
        self.stable_counter = 0
        self.smooth_cx = None
        self.smooth_cy = None

        if self.enable_payload_gpio:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(PAYLOAD_PIN, GPIO.OUT)
            GPIO.output(PAYLOAD_PIN, GPIO.LOW)

    def get_target_info(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        inp, r, dw, dh = preprocess(frame)
        output = self.trt_infer.infer(inp)
        detections = postprocess(output, frame, r, dw, dh)

        frame_h, frame_w = frame.shape[:2]
        image_cx = frame_w / 2.0
        image_cy = frame_h / 2.0

        if len(detections) == 0:
            self.prev_center = None
            self.prev_area = None
            self.stable_counter = 0
            self.smooth_cx = None
            self.smooth_cy = None

            return {
                "has_target": False,
                "confidence": 0.0,
                "x_error_px": 0.0,
                "y_error_px": 0.0,
                "stable": False,
                "stable_counter": 0
            }

        best_det = max(detections, key=lambda x: x[1])
        box, score, class_id = best_det
        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)

        if self.smooth_cx is None:
            self.smooth_cx = cx
            self.smooth_cy = cy
        else:
            self.smooth_cx = self.smooth_alpha * cx + (1.0 - self.smooth_alpha) * self.smooth_cx
            self.smooth_cy = self.smooth_alpha * cy + (1.0 - self.smooth_alpha) * self.smooth_cy

        if self.prev_center is not None and self.prev_area is not None:
            dist = ((cx - self.prev_center[0]) ** 2 + (cy - self.prev_center[1]) ** 2) ** 0.5
            area_ratio = area / max(self.prev_area, 1.0)

            if (
                score >= self.trigger_conf and
                dist < self.stable_dist_px and
                self.area_ratio_min < area_ratio < self.area_ratio_max
            ):
                self.stable_counter += 1
            else:
                self.stable_counter = max(0, self.stable_counter - 1)
        else:
            self.stable_counter = 1 if score >= self.trigger_conf else 0

        self.prev_center = (cx, cy)
        self.prev_area = area

        return {
            "has_target": True,
            "confidence": float(score),
            "x_error_px": self.smooth_cx - image_cx,
            "y_error_px": self.smooth_cy - image_cy,
            "stable": self.stable_counter >= 4,
            "stable_counter": self.stable_counter,
            "box": [int(x1), int(y1), int(x2), int(y2)]
        }

    def trigger_payload(self):
        if not self.enable_payload_gpio:
            print("Payload GPIO disabled")
            return
        print("TRIGGERING PAYLOAD")
        GPIO.output(PAYLOAD_PIN, GPIO.HIGH)
        time.sleep(PAYLOAD_PULSE_S)
        GPIO.output(PAYLOAD_PIN, GPIO.LOW)

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        if self.enable_payload_gpio:
            try:
                GPIO.output(PAYLOAD_PIN, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass