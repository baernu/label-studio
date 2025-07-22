


import os
import logging
from typing import List, Dict, Optional
import requests
from flask import Flask, request, jsonify, Response
import json
from label_studio_ml.model import LabelStudioMLBase

# Set up logging
log_file = os.path.join(os.getcwd(), "app.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(log_file)  # Logs to file
    ]
)
logger = logging.getLogger(__name__)


app = Flask(__name__)


class YOLOBackend(LabelStudioMLBase):
    """Label Studio ML Backend for YOLO."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.load_model()

    def load_model(self):
        """Load the YOLO model."""
        from ultralytics import YOLO
        #model_path = os.getenv("YOLO_MODEL_PATH", "retrained_4.pt")
        model_path = os.getenv("YOLO_MODEL_PATH", "runs/train35/weights/last.pt")
        logger.info(f"Loading YOLO model from {model_path}")
        return YOLO(model_path)



    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        """Run YOLO predictions and return results in Label Studio format."""
        logger.info(f"Received {len(tasks)} tasks for prediction.")
        predictions = []

        # Default hostname for image retrieval
        hostname = context["hostname"] if context and "hostname" in context else "http://172.22.0.3:8080"

        for idx, task in enumerate(tasks):
            logger.info(f"Processing task {idx + 1}/{len(tasks)} with task ID: {task.get('id', 'unknown')}")
            try:
                # Download the image
                image_url = task["data"]["image"]
                logger.info(f"Task {idx + 1}: Downloading image from {image_url}")
                image_path = self._download_image(image_url, hostname)

                # Run YOLO inference
                logger.info(f"Task {idx + 1}: Running YOLO inference on {image_path}")
                results = self.model(image_path, conf=0.25)
                prediction = results[0]

                # Process YOLO output
                regions = self.process_yolo_output(prediction)
                logger.info(f"Task {idx + 1}: Processed regions: {regions}")

                # Append predictions with correct structure
                predictions.append({
                    "model_version": "yolo-v8",
                    "result": regions,
                    "score": max([region.get("score", 0) for region in regions if "score" in region], default=0)
                })

            except Exception as e:
                logger.error(f"Error processing task {task.get('id', 'unknown')} at index {idx + 1}: {e}",
                             exc_info=True)
                # Handle failed task
                predictions.append({
                    "model_version": "yolo-v8",
                    "result": [],
                    "score": 0.0
                })

        logger.info(f"Final predictions: {predictions}")
        return {"predictions": predictions}


    def process_yolo_output(self, prediction):
        """Process YOLO output into Label Studio-compatible format."""
        regions = []

        # Get original image dimensions from prediction
        img_width = prediction.orig_shape[1]
        img_height = prediction.orig_shape[0]
        logger.debug(f"Image dimensions (width x height): {img_width} x {img_height}")



        # Process keypoints in the exact order of KEYPOINT_LABELS
        keypoints_order = [

            # Visible Keypoints
            "Teat Head Back Right - Visible", "Teat Origin Back Right - Visible",
            "Teat Head Back Left - Visible", "Teat Origin Back Left - Visible",
            "Teat Head Front Right - Visible", "Teat Origin Front Right - Visible",
            "Teat Head Front Left - Visible", "Teat Origin Front Left - Visible",
            "Udder Center Front - Visible", "Udder Center Back - Visible",
            "Shackle Right - Visible", "Leg Right - Visible",
            "Shackle Left - Visible", "Leg Left - Visible",
        ]

        if hasattr(prediction, "keypoints") and prediction.keypoints is not None:
            raw_keypoints = prediction.keypoints.xy.cpu().numpy()
            logger.debug(f"Raw keypoints: {raw_keypoints}")

            for i, label in enumerate(keypoints_order):
                if i < raw_keypoints.shape[1]:
                    # Convert pixel coordinates to normalized percentages
                    x_pixel, y_pixel = raw_keypoints[0, i, 0], raw_keypoints[0, i, 1]
                    x = (x_pixel / img_width) * 100
                    y = (y_pixel / img_height) * 100

                    if x == 0 and y == 0:
                        x, y = 50 + i, 96
                        label = label.replace("Visible", "Invisible")


                regions.append({
                    "from_name": "keypoints",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "value": {
                        "keypointlabels": [label],
                        "x": x,
                        "y": y,
                        "width": 0.8
                    }})

        # Process bounding boxes (if any)
        if hasattr(prediction, "boxes") and prediction.boxes is not None:
            for bbox, confidence in zip(prediction.boxes.xywhn, prediction.boxes.conf):
                x_center, y_center, w, h = bbox.cpu().numpy()
                # Convert center coordinates to top-left and normalize to percentages
                x = (x_center - w / 2) * 100  # Top-left x
                y = (y_center - h / 2) * 100  # Top-left y
                w, h = w * 100, h * 100  # Width and height in percentages
                regions.append({
                    "from_name": "bbox",
                    "image_rotation": 0,
                    "original_height": img_height,
                    "original_width": img_width,
                    "score": float(confidence.cpu().item()),
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": h,
                        "rectanglelabels": ["Cow-Udder"],
                        "width": w,
                        "x": x,
                        "y": y
                    },

                })

        return regions

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



# Define endpoints
backend = YOLOBackend()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "UP"})



@app.route("/setup", methods=["POST"])
def setup():
    """Setup endpoint for Label Studio."""
    data = request.json
    logger.info(f"Setup request received: {data}")

    # Validate and correct the project field
    if 'project' in data:
        try:
            data['project'] = int(float(data['project']))
            logger.info(f"Corrected project field to: {data['project']}")
        except ValueError:
            logger.error(f"Invalid project field: {data['project']}")

    return jsonify({"status": "OK"})




@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    try:
        tasks = request.json.get("tasks", [])
        context = request.json.get("context", {})
        logger.info(f"Incoming /predict request with {len(tasks)} tasks.")
        logger.debug(f"Request payload: {request.json}")

        # Get predictions from backend
        predictions = backend.predict(tasks, context)

        # Wrap predictions under the `results` key
        response = {"results": predictions.get("predictions", [])}

        # Log and return the response
        logger.info(f"Returning response for /predict: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error handling /predict request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)

