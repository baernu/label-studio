import os
import logging
from typing import List, Dict, Optional
import requests
from flask import Flask, request, jsonify
from label_studio_ml.model import LabelStudioMLBase
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import nms

# Set up logging
log_file = os.path.join(os.getcwd(), "app.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_grids(input_size, strides):
    coords = []
    for s in strides:
        gh, gw = input_size[0] // s, input_size[1] // s
        y, x = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
        coords.append(np.stack([x.reshape(-1), y.reshape(-1), np.full(x.size, s)], axis=-1))
    return np.vstack(coords)



# Debug helper
def debug_model_io(input_tensor, model_outputs, model_session):
    logger.debug(f"Input tensor shape: {input_tensor.shape}")
    logger.debug(f"Input tensor dtype: {input_tensor.dtype}")

    logger.debug("ONNX model input:")
    for i in model_session.get_inputs():
        logger.debug(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

    logger.debug("ONNX model output:")
    for o in model_session.get_outputs():
        logger.debug(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")

    logger.debug(f"Raw model output sample (first 3 rows):\n{model_outputs[0][:3]}")


def preprocess_yolox_image(image_path, input_size=(608, 1280)):
    img = cv2.imread(image_path)
    h0, w0 = img.shape[:2]
    scale = min(input_size[0] / h0, input_size[1] / w0)
    nh, nw = int(h0 * scale), int(w0 * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized

    img_input = np.transpose(padded, (2, 0, 1))[None].astype(np.float32)
    return img_input, scale, (w0, h0)

def xywh2xyxy(boxes):
    x, y, w, h = boxes.T
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

def decode_output(raw_output):
    """
    Decodes the raw model output.
    Assumes shape (N, 5 + num_classes): [x, y, w, h, obj_conf, cls_scores...]
    """
    xy = raw_output[:, :2]
    wh = raw_output[:, 2:4]
    obj_conf = sigmoid(raw_output[:, 4:5])
    cls_scores = sigmoid(raw_output[:, 5:])
    scores = obj_conf * cls_scores
    boxes = np.concatenate([xy, wh], axis=1)
    return boxes, scores


def postprocess_yolox_output(boxes, scores, conf_thres, nms_thres, class_names, scale, orig_size, input_size):
    """
    Applies confidence threshold and NMS, then formats predictions for Label Studio.
    """
    cls_ids = np.argmax(scores, axis=1)
    cls_conf = scores[np.arange(len(scores)), cls_ids]
    mask = cls_conf > conf_thres

    if not np.any(mask):
        return []

    boxes = boxes[mask]
    cls_ids = cls_ids[mask]
    cls_conf = cls_conf[mask]
    bboxes_xyxy = xywh2xyxy(boxes)

    # Torch NMS
    keep = nms(
        torch.tensor(bboxes_xyxy, dtype=torch.float32),
        torch.tensor(cls_conf, dtype=torch.float32),
        nms_thres
    ).numpy()

    w_orig, h_orig = orig_size
    results = []
    # for i in keep:
    #     x1, y1, x2, y2 = bboxes_xyxy[i]
    #
    #     # ✅ Properly map back to original image size using the resize scale
    #     x1 = x1 / scale
    #     x2 = x2 / scale
    #     y1 = y1 / scale
    #     y2 = y2 / scale
    #
    #     # ✅ Clip to original image dimensions
    #     # x1 = np.clip(x1, 0, w_orig)
    #     # x2 = np.clip(x2, 0, w_orig)
    #     # y1 = np.clip(y1, 0, h_orig)
    #     # y2 = np.clip(y2, 0, h_orig)
    #
    #     width = x2 - x1
    #     height = y2 - y1
    #
    #     results.append({
    #         "from_name": "label",
    #         "to_name": "image",
    #         "type": "rectanglelabels",
    #         "original_width": w_orig,
    #         "original_height": h_orig,
    #         "image_rotation": 0,
    #         "score": float(cls_conf[i]),
    #         "value": {
    #             "x": x1 / w_orig * 100,
    #             "y": y1 / h_orig * 100,
    #             "width": width / w_orig * 100,
    #             "height": height / h_orig * 100,
    #             "rectanglelabels": [class_names[cls_ids[i]]]
    #         }
    #     })


    for i in keep:
        x1, y1, x2, y2 = bboxes_xyxy[i]

        x1 = x1 / scale
        x2 = x2 / scale
        y1 = y1 / scale
        y2 = y2 / scale

        width = x2 - x1
        height = y2 - y1

        cls_id = cls_ids[i]
        label = class_names[cls_id]

        logger.debug(f"[POST] Box {i}: class_id={cls_id}, label={label}, conf={cls_conf[i]:.3f}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        # 🧪 Safety check
        if label not in class_names:
            logger.error(f"[MISMATCH] Label '{label}' not in class_names list! (Index {cls_id})")

        result = {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": w_orig,
            "original_height": h_orig,
            "image_rotation": 0,
            "score": float(cls_conf[i]),
            "value": {
                "x": x1 / w_orig * 100,
                "y": y1 / h_orig * 100,
                "width": width / w_orig * 100,
                "height": height / h_orig * 100,
                "rectanglelabels": [label]
            }
        }

        logger.debug(f"[POST] Final result payload: {result}")
        results.append(result)

    return results



class YOLOBackend(LabelStudioMLBase):
    """Label Studio ML Backend for YOLOX ONNX"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.class_names = list("SWTZUOPHJKBXYR")
        self.class_names = [
            "B", "H", "J", "K", "O", "P", "R",
            "S", "T", "U", "W", "X", "Y", "Z"
        ]
        self.model = self.load_model()

    def load_model(self):
        model_path = os.getenv("YOLOX_MODEL_PATH", "letters_yolox_fewshot1.6.onnx")
        logger.info(f"Loading YOLOX ONNX model from {model_path}")
        return ort.InferenceSession(model_path)


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        logger.info(f"Received {len(tasks)} tasks for YOLOX prediction.")
        predictions = []

        import subprocess

        def get_docker_gateway():
            try:
                result = subprocess.check_output("ip route | grep default", shell=True).decode()
                gateway_ip = result.strip().split(" ")[2]
                return gateway_ip
            except Exception as e:
                return "172.17.0.1"  # fallback

        hostname = context.get("hostname", f"http://{get_docker_gateway()}:8080")


        for idx, task in enumerate(tasks):
            try:
                image_url = task["data"]["image"]
                logger.info(f"Downloading image from task {idx + 1}: {image_url}")
                image_path = self._download_image(image_url, hostname)

                # input_tensor, orig_w, orig_h = preprocess_yolox_image(image_path)
                input_tensor, scale, (orig_w, orig_h) = preprocess_yolox_image(image_path)
                logger.debug(f"Original image size: {orig_w}x{orig_h}")

                ort_inputs = {self.model.get_inputs()[0].name: input_tensor}
                ort_outs = self.model.run(None, ort_inputs)

                debug_model_io(input_tensor, ort_outs, self.model)

                output = ort_outs[0][0]
                boxes, scores = decode_output(output)
                regions = postprocess_yolox_output(
                    boxes, scores,
                    conf_thres=0.4,
                    nms_thres=0.45,
                    class_names=self.class_names,
                    scale=scale,
                    orig_size=(orig_w, orig_h),
                    input_size=(608, 1280)  # or define as self.input_size if class-based
                )

                # for r in regions:
                #     print(
                #         f"[DEBUG] Detection: {r['value']['rectanglelabels'][0]} @ {r['value']} score={r['score']:.2f}")

                if not regions:
                    logger.warning(f"No detections found for task {task.get('id')}")



                predictions.append({
                    "model_version": "yolox-onnx",
                    "result": regions,
                    "score": max([r.get("score", 0) for r in regions], default=0)
                })

            except Exception as e:
                logger.exception(f"Failed to process task {task.get('id')}: {e}")
                predictions.append({
                    "model_version": "yolox-onnx",
                    "result": [],
                    "score": 0.0
                })

        return {"predictions": predictions}


    def _download_image(self, url: str, hostname: str) -> str:
        """Download an image from a URL."""
        try:

            if not url.startswith("http"):
                url = f"{hostname.rstrip('/')}/{url.lstrip('/')}"

            logger.info(f"Downloading image from: {url}")
            headers = {"Authorization": "Token 676c739104a47a882f6e3596b28f9efa75cf9395"}

            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 200:
                image_path = f"/tmp/{os.path.basename(url)}"
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return image_path

            logger.error(f"Failed to download image from {url}. Status code: {response.status_code}")
            raise ValueError(f"Failed to download image from {url}, status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception occurred while downloading image from {url}: {e}")
            raise


# Flask endpoints
backend = YOLOBackend()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "YOLOX backend is up"})


@app.route("/setup", methods=["POST"])
def setup():
    data = request.json
    logger.info(f"Setup received: {data}")
    if 'project' in data:
        try:
            data['project'] = int(float(data['project']))
        except ValueError:
            logger.error("Invalid project field")
    return jsonify({"status": "OK"})





@app.route("/predict", methods=["POST"])
def predict():
    try:
        tasks = request.json.get("tasks", [])
        context = request.json.get("context", {})
        logger.info(f"/predict request: {len(tasks)} tasks")
        predictions = backend.predict(tasks, context)
        return jsonify({"results": predictions.get("predictions", [])})
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
