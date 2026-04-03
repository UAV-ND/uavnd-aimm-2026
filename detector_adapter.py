#!/usr/bin/env python3
# detector_adapter.py
#
# Unified detector for a single TensorRT model with three classes:
#   "black_buoy", "target", "boat"
#
# Each mission stage constructs GenericDetector with the class it cares about.
# Detections that don't match target_class are ignored entirely.

import time
import cv2
import threading
import Jetson.GPIO as GPIO

from minimal_cv import TensorRTInfer, preprocess, postprocess, gstreamer_pipeline

VALID_CLASSES = ["black_buoy", "target", "boat"]

PAYLOAD_PIN = 16
PAYLOAD_PULSE_S = 0.5


class GenericDetector:
    def __init__(
        self,
        engine_path,
        target_class,                  # "black_buoy" | "target" | "boat"
        smooth_alpha=0.12,
        trigger_conf=0.97,
        stable_dist_px=100,
        area_ratio_min=0.55,
        area_ratio_max=1.80,
        enable_payload_gpio=False
    ):
        if target_class not in VALID_CLASSES:
            raise ValueError(
                "target_class '{}' is not valid. Choose from: {}".format(
                    target_class, VALID_CLASSES
                )
            )

        self.target_class = target_class
        self.smooth_alpha = smooth_alpha
        self.trigger_conf = trigger_conf
        self.stable_dist_px = stable_dist_px
        self.area_ratio_min = area_ratio_min
        self.area_ratio_max = area_ratio_max
        self.enable_payload_gpio = enable_payload_gpio

        # Class index is the position in VALID_CLASSES.
        # This must match the order your model was trained with.
        self.target_class_id = VALID_CLASSES.index(target_class)

        print("[GenericDetector] Loading TensorRT engine: {}".format(engine_path))
        print("[GenericDetector] Filtering for class: '{}' (id={})".format(
            target_class, self.target_class_id
        ))
        self.trt_infer = TensorRTInfer(engine_path)

        pipeline = gstreamer_pipeline(flip_method=0)
        print("[GenericDetector] Pipeline: {}".format(pipeline))

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
        """
        Returns a dict:
          has_target      bool
          confidence      float
          x_error_px      float  (positive = target is RIGHT of center)
          y_error_px      float  (positive = target is BELOW center)
          stable          bool   (True once stable_counter >= 4)
          stable_counter  int
          box             [x1, y1, x2, y2] or absent if no target
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        inp, r, dw, dh = preprocess(frame)
        output = self.trt_infer.infer(inp)
        all_detections = postprocess(output, frame, r, dw, dh)

        # Filter to only the class this detector instance cares about
        detections = [
            (box, score, class_id)
            for box, score, class_id in all_detections
            if class_id == self.target_class_id
        ]

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

        # Pick highest confidence detection for this class
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
            dist = (
                (cx - self.prev_center[0]) ** 2 + (cy - self.prev_center[1]) ** 2
            ) ** 0.5
            area_ratio = area / max(self.prev_area, 1.0)

            if (
                score >= self.trigger_conf
                and dist < self.stable_dist_px
                and self.area_ratio_min < area_ratio < self.area_ratio_max
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
        """Non-blocking GPIO pulse to release payload."""
        if not self.enable_payload_gpio:
            print("[GenericDetector] Payload GPIO disabled — dry run only")
            return

        def _pulse():
            print("[GenericDetector] TRIGGERING PAYLOAD on pin {}".format(PAYLOAD_PIN))
            GPIO.output(PAYLOAD_PIN, GPIO.HIGH)
            time.sleep(PAYLOAD_PULSE_S)
            GPIO.output(PAYLOAD_PIN, GPIO.LOW)

        threading.Thread(target=_pulse, daemon=True).start()

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