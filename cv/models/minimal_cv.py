#!/usr/bin/env python3

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import Jetson.GPIO as GPIO
import time
import threading

ENGINE_PATH = "engine.engine"
CLASS_NAMES = ["black_bottle"]

INPUT_W = 640
INPUT_H = 640

CONF_THRES = 0.93
NMS_THRES = 0.35

TRIGGER_CONF = 0.97
TRIGGER_FRAMES = 8

MIN_BOX_W = 50
MIN_BOX_H = 50
MIN_BOX_AREA = 3000

CENTER_GATE_X = 0.48
CENTER_GATE_Y = 0.48

STABLE_DIST_PX = 100
AREA_RATIO_MIN = 0.55
AREA_RATIO_MAX = 1.80

SMOOTH_ALPHA = 0.12

PAYLOAD_PIN = 16
PAYLOAD_PULSE_S = 2.0
TRIGGER_COOLDOWN_S = 3.0
ENABLE_GPIO_TRIGGER = True

DETECT_EVERY_N_FRAMES = 3
MAX_LOST_FRAMES = 6

FLOW_MIN_POINTS = 6
FLOW_MAX_CORNERS = 60
FLOW_QUALITY_LEVEL = 0.2
FLOW_MIN_DISTANCE = 7
FLOW_BLOCK_SIZE = 7
FLOW_MAX_JUMP_PX = 160

TRACKER_MIN_BOX_W = 40
TRACKER_MIN_BOX_H = 40


def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PAYLOAD_PIN, GPIO.OUT)
    GPIO.output(PAYLOAD_PIN, GPIO.LOW)


def cleanup_gpio():
    try:
        GPIO.output(PAYLOAD_PIN, GPIO.LOW)
    except Exception:
        pass
    GPIO.cleanup()

def trigger_payload(pin=PAYLOAD_PIN, pulse_s=PAYLOAD_PULSE_S):
    def _pulse():
        print("TRIGGERING PAYLOAD on pin", pin)
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(pulse_s)
        GPIO.output(pin, GPIO.LOW)
    threading.Thread(target=_pulse, daemon=True).start()

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=25,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(float(new_shape[0]) / h, float(new_shape[1]) / w)

    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, dw, dh


def preprocess(frame):
    img, r, dw, dh = letterbox(frame, (INPUT_H, INPUT_W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    return img, r, dw, dh


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def postprocess(output, orig_img, r, dw, dh):
    h0, w0 = orig_img.shape[:2]
    img_cx = w0 / 2.0
    img_cy = h0 / 2.0

    preds = output.reshape(1, 5, 8400)[0]
    preds = preds.T

    boxes = []
    scores = []
    class_ids = []

    for row in preds:
        cx, cy, w, h = row[:4]
        class_scores = row[4:]

        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])

        if score < CONF_THRES:
            continue

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r

        x1 = max(0, min(x1, w0 - 1))
        y1 = max(0, min(y1, h0 - 1))
        x2 = max(0, min(x2, w0 - 1))
        y2 = max(0, min(y2, h0 - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue
        if area < MIN_BOX_AREA:
            continue

        box_cx = (x1 + x2) / 2.0
        box_cy = (y1 + y2) / 2.0

        if abs(box_cx - img_cx) > CENTER_GATE_X * w0:
            continue
        if abs(box_cy - img_cy) > CENTER_GATE_Y * h0:
            continue

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    keep = nms(boxes, scores, NMS_THRES)
    return [(boxes[i], scores[i], class_ids[i]) for i in keep]


def clip_box_xyxy(box, frame_shape):
    h0, w0 = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w0 - 1))
    y1 = max(0, min(y1, h0 - 1))
    x2 = max(0, min(x2, w0 - 1))
    y2 = max(0, min(y2, h0 - 1))
    return [float(x1), float(y1), float(x2), float(y2)]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def box_center_area(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return cx, cy, area


def is_reasonable_box(box, frame_shape):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if x2 <= x1 or y2 <= y1:
        return False
    if bw < TRACKER_MIN_BOX_W or bh < TRACKER_MIN_BOX_H:
        return False

    h0, w0 = frame_shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if abs(cx - (w0 / 2.0)) > 0.58 * w0:
        return False
    if abs(cy - (h0 / 2.0)) > 0.58 * h0:
        return False

    return True


class OpticalFlowTracker:
    def __init__(self):
        self.prev_gray = None
        self.points = None
        self.box_xyxy = None

    def init(self, frame, box_xyxy):
        self.box_xyxy = [float(v) for v in box_xyxy]
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = self._extract_points(self.prev_gray, self.box_xyxy)

    def _extract_points(self, gray, box_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1] - 1, x2)
        y2 = min(gray.shape[0] - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        p0 = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=FLOW_MAX_CORNERS,
            qualityLevel=FLOW_QUALITY_LEVEL,
            minDistance=FLOW_MIN_DISTANCE,
            blockSize=FLOW_BLOCK_SIZE,
        )

        if p0 is None:
            return None

        p0[:, 0, 0] += x1
        p0[:, 0, 1] += y1
        return p0

    def refresh_points(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.points = self._extract_points(gray, self.box_xyxy)

    def update(self, frame):
        if self.prev_gray is None or self.points is None:
            return False, None

        if len(self.points) < FLOW_MIN_POINTS:
            self.refresh_points(frame)
            if self.points is None or len(self.points) < FLOW_MIN_POINTS:
                return False, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        if p1 is None or st is None:
            return False, None

        good_new = p1[st == 1]
        good_old = self.points[st == 1]

        if len(good_new) < FLOW_MIN_POINTS:
            self.prev_gray = gray
            self.points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
            return False, None

        dx = float(np.median(good_new[:, 0] - good_old[:, 0]))
        dy = float(np.median(good_new[:, 1] - good_old[:, 1]))

        x1, y1, x2, y2 = self.box_xyxy
        new_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

        self.prev_gray = gray
        self.points = good_new.reshape(-1, 1, 2)
        self.box_xyxy = new_box

        return True, new_box


class TensorRTInfer(object):
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = [None] * self.engine.num_bindings
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []

        for i in range(self.engine.num_bindings):
            print(
                "binding", i,
                "name=", self.engine.get_binding_name(i),
                "shape=", self.engine.get_binding_shape(i),
                "dtype=", self.engine.get_binding_dtype(i),
                "is_input=", self.engine.binding_is_input(i)
            )

            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings[i] = int(cuda_mem)

            if self.engine.binding_is_input(i):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def infer(self, input_image):
        np.copyto(self.host_inputs[0], input_image.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        return self.host_outputs[0]


def main():
    print("Loading TensorRT engine:", ENGINE_PATH)
    trt_infer = TensorRTInfer(ENGINE_PATH)

    if ENABLE_GPIO_TRIGGER:
        setup_gpio()

    pipeline = gstreamer_pipeline(flip_method=0)
    print("Using pipeline:")
    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    prev_center = None
    prev_area = None
    stable_counter = 0

    smooth_cx = None
    smooth_cy = None

    last_trigger_time = 0.0

    tracker = None
    tracker_active = False
    tracker_box_xyxy = None
    tracker_class_id = 0
    tracker_score = 0.0

    lost_frames = 0
    frame_idx = 0

    print("Running headless — press Ctrl+C to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_idx += 1
        best_det = None
        best_score = 0.0
        mode_text = "TRACK" if tracker_active else "SEARCH"
        used_tracker_this_frame = False

        if tracker_active and tracker is not None:
            ok, tracked_box_xyxy = tracker.update(frame)

            if ok:
                tracked_box_xyxy = clip_box_xyxy(tracked_box_xyxy, frame.shape)

                if is_reasonable_box(tracked_box_xyxy, frame.shape):
                    if tracker_box_xyxy is not None:
                        cx_new, cy_new, _ = box_center_area(tracked_box_xyxy)
                        cx_old, cy_old, _ = box_center_area(tracker_box_xyxy)
                        jump = ((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2) ** 0.5

                        if jump <= FLOW_MAX_JUMP_PX:
                            tracker_box_xyxy = tracked_box_xyxy
                            tracker.box_xyxy = tracker_box_xyxy
                            best_det = (tracker_box_xyxy, tracker_score, tracker_class_id)
                            lost_frames = 0
                            used_tracker_this_frame = True
                        else:
                            lost_frames += 1
                    else:
                        tracker_box_xyxy = tracked_box_xyxy
                        tracker.box_xyxy = tracker_box_xyxy
                        best_det = (tracker_box_xyxy, tracker_score, tracker_class_id)
                        lost_frames = 0
                        used_tracker_this_frame = True
                else:
                    lost_frames += 1
            else:
                lost_frames += 1

            if lost_frames > MAX_LOST_FRAMES:
                print("Tracker lost target; returning to SEARCH")
                tracker = None
                tracker_active = False
                tracker_box_xyxy = None
                tracker_score = 0.0
                lost_frames = 0
                stable_counter = 0
                prev_center = None
                prev_area = None
                smooth_cx = None
                smooth_cy = None
                mode_text = "SEARCH"

        should_detect = (not tracker_active) or (frame_idx % DETECT_EVERY_N_FRAMES == 0)

        if should_detect:
            inp, r, dw, dh = preprocess(frame)
            output = trt_infer.infer(inp)
            detections = postprocess(output, frame, r, dw, dh)

            for box, score, class_id in detections:
                if score > best_score:
                    best_score = score
                    best_det = (box, score, class_id)

            if best_det is not None:
                det_box, det_score, det_class_id = best_det

                if not tracker_active:
                    tracker = OpticalFlowTracker()
                    tracker.init(frame, det_box)

                    if tracker.points is not None and len(tracker.points) >= FLOW_MIN_POINTS:
                        tracker_active = True
                        tracker_box_xyxy = det_box
                        tracker_score = det_score
                        tracker_class_id = det_class_id
                        lost_frames = 0
                        mode_text = "DETECT+FLOW"
                        print("Acquired target from detector")
                    else:
                        tracker = None
                        tracker_active = False
                        tracker_box_xyxy = det_box
                        tracker_score = det_score
                        tracker_class_id = det_class_id
                        mode_text = "DETECT_ONLY"
                else:
                    tracker = OpticalFlowTracker()
                    tracker.init(frame, det_box)

                    if tracker.points is not None and len(tracker.points) >= FLOW_MIN_POINTS:
                        tracker_box_xyxy = det_box
                        tracker_score = det_score
                        tracker_class_id = det_class_id
                        lost_frames = 0
                        mode_text = "REFRESH_FLOW"
                    else:
                        tracker = None
                        tracker_active = False
                        tracker_box_xyxy = det_box
                        tracker_score = det_score
                        tracker_class_id = det_class_id
                        mode_text = "DETECT_ONLY"

                used_tracker_this_frame = False
            else:
                if not tracker_active:
                    best_det = None
                    mode_text = "SEARCH"

        if best_det is not None:
            box, score, class_id = best_det
            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)

            if smooth_cx is None:
                smooth_cx, smooth_cy = cx, cy
            else:
                smooth_cx = SMOOTH_ALPHA * cx + (1.0 - SMOOTH_ALPHA) * smooth_cx
                smooth_cy = SMOOTH_ALPHA * cy + (1.0 - SMOOTH_ALPHA) * smooth_cy

            if prev_center is not None and prev_area is not None:
                dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                area_ratio = area / max(prev_area, 1.0)

                effective_trigger_conf = TRIGGER_CONF if not used_tracker_this_frame else (TRIGGER_CONF - 0.02)

                if (
                    score >= effective_trigger_conf
                    and dist < STABLE_DIST_PX
                    and AREA_RATIO_MIN < area_ratio < AREA_RATIO_MAX
                ):
                    stable_counter += 1
                else:
                    stable_counter = max(0, stable_counter - 1)
            else:
                stable_counter = 1 if score >= TRIGGER_CONF else 0

            prev_center = (cx, cy)
            prev_area = area

            print(
                "TARGET:",
                CLASS_NAMES[class_id],
                "conf=", round(score, 3),
                "box=", [int(x1), int(y1), int(x2), int(y2)],
                "stable=", stable_counter,
                "tracker=", tracker_active,
                "lost=", lost_frames,
                "mode=", mode_text
            )
        else:
            stable_counter = max(0, stable_counter - 1)
            prev_center = None
            prev_area = None
            smooth_cx = None
            smooth_cy = None

        now = time.time()
        if stable_counter >= TRIGGER_FRAMES and (now - last_trigger_time) > TRIGGER_COOLDOWN_S:
            print("CONFIRMED TARGET EVENT")

            if ENABLE_GPIO_TRIGGER:
                trigger_payload()
            else:
                print("GPIO trigger disabled; dry run only")

            last_trigger_time = now
            stable_counter = 0

    cap.release()
    if ENABLE_GPIO_TRIGGER:
        cleanup_gpio()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
        if ENABLE_GPIO_TRIGGER:
            cleanup_gpio()