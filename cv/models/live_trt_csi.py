import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "best_fp16.engine"
CLASS_NAMES = ["Black buoy", "Target"]

INPUT_W = 640
INPUT_H = 640

CONF_THRES = 0.40
NMS_THRES = 0.45

TRIGGER_CONF = 0.70
TRIGGER_FRAMES = 5


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
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # CHW -> NCHW
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

    preds = output.reshape(1, 6, 8400)[0]   # (6, 8400)
    preds = preds.T                         # (8400, 6)

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

        # undo letterbox
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

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    keep = nms(boxes, scores, NMS_THRES)

    detections = []
    for i in keep:
        detections.append((boxes[i], scores[i], class_ids[i]))

    return detections


def draw_detections(image, detections):
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
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        return self.host_outputs[0]


def main():
    print("Loading TensorRT engine:", ENGINE_PATH)
    trt_infer = TensorRTInfer(ENGINE_PATH)

    pipeline = gstreamer_pipeline(flip_method=0)
    print("Using pipeline:")
    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Unable to open camera")
        return

    cv2.namedWindow("CSI TensorRT Detection", cv2.WINDOW_AUTOSIZE)

    target_counter = 0

    while cv2.getWindowProperty("CSI TensorRT Detection", 0) >= 0:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        inp, r, dw, dh = preprocess(frame)
        output = trt_infer.infer(inp)
        detections = postprocess(output, frame, r, dw, dh)

        confirmed = False
        for box, score, class_id in detections:
            if score >= TRIGGER_CONF:
                confirmed = True
                print("DETECTED:", CLASS_NAMES[class_id], "conf=", round(score, 3))

        if confirmed:
            target_counter += 1
        else:
            target_counter = 0

        if target_counter >= TRIGGER_FRAMES:
            print("CONFIRMED TARGET EVENT")
            # TODO: trigger payload here
            target_counter = 0

        vis = draw_detections(frame.copy(), detections)
        cv2.imshow("CSI TensorRT Detection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
        cv2.destroyAllWindows()
