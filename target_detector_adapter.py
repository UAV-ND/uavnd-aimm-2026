#!/usr/bin/env python3
import time
import cv2
import Jetson.GPIO as GPIO

from cv_2 import TensorRTInfer, preprocess, postprocess, gstreamer_pipeline

ENGINE_PATH = "engine.engine"

SMOOTH_ALPHA = 0.12
TRIGGER_CONF = 0.97
STABLE_DIST_PX = 100
AREA_RATIO_MIN = 0.55
AREA_RATIO_MAX = 1.80

PAYLOAD_PIN = 16
PAYLOAD_PULSE_S = 0.5


class TargetDetector:
    def __init__(self):
        print("Loading TensorRT engine:", ENGINE_PATH)
        self.trt_infer = TensorRTInfer(ENGINE_PATH)

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
            self.smooth_cx = SMOOTH_ALPHA * cx + (1.0 - SMOOTH_ALPHA) * self.smooth_cx
            self.smooth_cy = SMOOTH_ALPHA * cy + (1.0 - SMOOTH_ALPHA) * self.smooth_cy

        if self.prev_center is not None and self.prev_area is not None:
            dist = ((cx - self.prev_center[0]) ** 2 + (cy - self.prev_center[1]) ** 2) ** 0.5
            area_ratio = area / max(self.prev_area, 1.0)

            if (
                score >= TRIGGER_CONF and
                dist < STABLE_DIST_PX and
                AREA_RATIO_MIN < area_ratio < AREA_RATIO_MAX
            ):
                self.stable_counter += 1
            else:
                self.stable_counter = max(0, self.stable_counter - 1)
        else:
            self.stable_counter = 1 if score >= TRIGGER_CONF else 0

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

        try:
            GPIO.output(PAYLOAD_PIN, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass