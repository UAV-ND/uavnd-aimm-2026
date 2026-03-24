#!/usr/bin/env python3

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import Jetson.GPIO as GPIO
import time

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

STABLE_DIST_PX = 80
AREA_RATIO_MIN = 0.60
AREA_RATIO_MAX = 1.60

SMOOTH_ALPHA = 0.12

PAYLOAD_PIN = 16
PAYLOAD_PULSE_S = 0.5
TRIGGER_COOLDOWN_S = 3.0
ENABLE_GPIO_TRIGGER = True

# Motion robustness settings
DETECT_EVERY_N_FRAMES = 3
MAX_LOST_FRAMES = 6
TRACKER_MIN_BOX_W = 40
TRACKER_MIN_BOX_H = 40
TRACKER_MAX_JUMP_PX = 140


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
    print("TRIGGERING PAYLOAD on pin", pin)
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(pulse_s)
    GPIO.output(pin, GPIO.LOW)


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
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
        "video/x-raw, format=(string)BGR ! appsink"
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


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def clip_box_xyxy(box, frame_shape):
    h0, w0 = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w0 - 1))
    y1 = max(0, min(y1, h0 - 1))
    x2 = max(0, min(x2, w0 - 1))
    y2 = max(0, min(y2, h0 - 1))
    return [x1, y1, x2, y2]


def box_center_area(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return cx, cy, area


def is_reasonable_tracked_box(box, frame_shape):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw < TRACKER_MIN_BOX_W or bh < TRACKER_MIN_BOX_H:
        return False
    if x2 <= x1 or y2 <= y1:
        return False

    h0, w0 = frame_shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if abs(cx - (w0 / 2.0)) > 0.55 * w0:
        return False
    if abs(cy - (h0 / 2.0)) > 0.55 * h0:
        return False

    return True


def create_tracker():
    # Prefer CSRT; fallback to KCF depending on OpenCV build
    if hasattr(cv2, "TrackerCSRT_create"):
        print("Using OpenCV CSRT tracker")
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        print("Using OpenCV legacy CSRT tracker")
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, "TrackerKCF_create"):
        print("Using OpenCV KCF tracker")
        return cv2.TrackerKCF_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        print("Using OpenCV legacy KCF tracker")
        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError("No supported OpenCV tracker found (CSRT/KCF unavailable)")


def draw_detections(
    image,
    detections,
    stable_counter=0,
    smooth_center=None,
    mode_text="SEARCH",
    lost_frames=0,
):
    for box, score, class_id in detections:
        x1, y1, x2, y2 = [int(v) for v in box]
        label = "{} {:.2f}".format(CLASS_NAMES[class_id], score)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        image,
        "mode={}".format(mode_text),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        image,
        "stable_counter={}".format(stable_counter),
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        image,
        "lost_frames={}".format(lost_frames),
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2,
    )

    if smooth_center is not None:
        scx, scy = int(smooth_center[0]), int(smooth_center[1])
        cv2.circle(image, (scx, scy), 6, (0, 0, 255), -1)

    return image


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

    cv2.namedWindow("CSI TensorRT Detection", cv2.WINDOW_AUTOSIZE)

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

    while cv2.getWindowProperty("CSI TensorRT Detection", 0) >= 0:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_idx += 1

        best_det = None
        best_score = 0.0
        mode_text = "TRACK" if tracker_active else "SEARCH"
        used_tracker_this_frame = False

        # 1) Tracker update every frame if active
        if tracker_active and tracker is not None:
            ok, tracked_box_xywh = tracker.update(frame)

            if ok:
                tracked_box_xyxy = clip_box_xyxy(
                    xywh_to_xyxy(tracked_box_xywh),
                    frame.shape
                )

                if is_reasonable_tracked_box(tracked_box_xyxy, frame.shape):
                    if tracker_box_xyxy is not None:
                        cx_new, cy_new, _ = box_center_area(tracked_box_xyxy)
                        cx_old, cy_old, _ = box_center_area(tracker_box_xyxy)
                        jump = ((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2) ** 0.5
                        if jump <= TRACKER_MAX_JUMP_PX:
                            tracker_box_xyxy = tracked_box_xyxy
                            best_det = (tracker_box_xyxy, tracker_score, tracker_class_id)
                            lost_frames = 0
                            used_tracker_this_frame = True
                        else:
                            lost_frames += 1
                    else:
                        tracker_box_xyxy = tracked_box_xyxy
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

        # 2) Periodic detector update, or always when tracker is inactive
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

                # If tracker is inactive, acquire target
                if not tracker_active:
                    tracker = create_tracker()
                    tracker.init(frame, xyxy_to_xywh(det_box))
                    tracker_active = True
                    tracker_box_xyxy = det_box
                    tracker_score = det_score
                    tracker_class_id = det_class_id
                    lost_frames = 0
                    mode_text = "DETECT+TRACK"
                    print("Acquired target from detector")
                else:
                    # If tracker already active, detector refreshes the tracker
                    tracker = create_tracker()
                    tracker.init(frame, xyxy_to_xywh(det_box))
                    tracker_box_xyxy = det_box
                    tracker_score = det_score
                    tracker_class_id = det_class_id
                    lost_frames = 0
                    mode_text = "REFRESH_TRACK"

                used_tracker_this_frame = False
            else:
                if not tracker_active:
                    best_det = None
                    mode_text = "SEARCH"

        # Build display detections
        display_detections = [best_det] if best_det is not None else []

        smooth_center = None

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

            smooth_center = (smooth_cx, smooth_cy)

            if prev_center is not None and prev_area is not None:
                dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                area_ratio = area / max(prev_area, 1.0)

                # For tracker frames, allow a little more tolerance
                effective_trigger_conf = TRIGGER_CONF if not used_tracker_this_frame else (TRIGGER_CONF - 0.02)

                if score >= effective_trigger_conf and dist < STABLE_DIST_PX and AREA_RATIO_MIN < area_ratio < AREA_RATIO_MAX:
                    stable_counter += 1
                else:
                    # decay instead of hard reset for motion tolerance
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

        vis = draw_detections(
            frame.copy(),
            display_detections,
            stable_counter=stable_counter,
            smooth_center=smooth_center,
            mode_text=mode_text,
            lost_frames=lost_frames,
        )
        cv2.imshow("CSI TensorRT Detection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if ENABLE_GPIO_TRIGGER:
        cleanup_gpio()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
        cv2.destroyAllWindows()
        if ENABLE_GPIO_TRIGGER:
            cleanup_gpio()
