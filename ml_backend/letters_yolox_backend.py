import os
import logging
from typing import List, Dict, Optional
import requests
from flask import Flask, request, jsonify
from label_studio_ml.model import LabelStudioMLBase
import cv2
import numpy as np
import onnxruntime as ort

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


def preprocess_yolox_image(image_path, input_size=(608, 1280)):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (input_size[1], input_size[0]))  # (W, H)
    img_input = img_resized[:, :, ::-1].astype(np.float32)  # BGR to RGB
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0) / 255.0
    return img_input.astype(np.float32), img.shape[1], img.shape[0]  # input, orig W, H


def parse_yolox_output(output, conf_thresh, img_width, img_height, class_names):
    detections = []
    pred = output[0]  # shape: [N, 5+num_classes]

    for row in pred:
        x, y, w, h = row[:4]
        obj_conf = sigmoid(row[4])
        class_scores = row[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        conf = obj_conf * class_conf

        if conf < conf_thresh:
            continue

        # Convert to top-left format and normalize
        x1 = (x - w / 2) / img_width
        y1 = (y - h / 2) / img_height
        w /= img_width
        h /= img_height

        detections.append({
            "from_name": "label",
            "image_rotation": 0,
            "original_height": img_height,
            "original_width": img_width,
            "score": float(conf),
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x1 * 100,
                "y": y1 * 100,
                "width": w * 100,
                "height": h * 100,
                "rectanglelabels": [class_names[class_id]]
            }
        })

    return detections


class YOLOBackend(LabelStudioMLBase):
    """Label Studio ML Backend for YOLOX ONNX"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_names = list("SWTZUOPHJKBXYR")
        self.model = self.load_model()

    def load_model(self):
        model_path = os.getenv("YOLOX_MODEL_PATH", "letters_yolox_608nano3.2.onnx")
        logger.info(f"Loading YOLOX ONNX model from {model_path}")
        return ort.InferenceSession(model_path)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        logger.info(f"Received {len(tasks)} tasks for YOLOX prediction.")
        predictions = []
        hostname = context.get("hostname", "http://172.22.0.3:8080") if context else "http://172.22.0.3:8080"

        for idx, task in enumerate(tasks):
            try:
                image_url = task["data"]["image"]
                logger.info(f"Downloading image from task {idx + 1}: {image_url}")
                image_path = self._download_image(image_url, hostname)

                input_tensor, orig_w, orig_h = preprocess_yolox_image(image_path)
                ort_inputs = {self.model.get_inputs()[0].name: input_tensor}
                ort_outs = self.model.run(None, ort_inputs)

                regions = parse_yolox_output(ort_outs, conf_thresh=0.25,
                                             img_width=orig_w, img_height=orig_h,
                                             class_names=self.class_names)

                predictions.append({
                    "model_version": "yolox-onnx",
                    "result": regions,
                    "score": max([r.get("score", 0) for r in regions], default=0)
                })

            except Exception as e:
                logger.exception(f"Failed to process task: {e}")
                predictions.append({
                    "model_version": "yolox-onnx",
                    "result": [],
                    "score": 0.0
                })

        return {"predictions": predictions}

    def _download_image(self, url: str, hostname: str) -> str:
        if not url.startswith("http"):
            url = f"{hostname.rstrip('/')}/{url.lstrip('/')}"
        logger.info(f"Resolved image URL: {url}")
        headers = {"Authorization": "Token 676c739104a47a882f6e3596b28f9efa75cf9395"}

        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            image_path = f"/tmp/{os.path.basename(url)}"
            with open(image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return image_path
        raise ValueError(f"Failed to download image from {url}, status code: {response.status_code}")


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
