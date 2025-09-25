import requests
from PIL import Image, ImageDraw, ImageFont
import io

# === CONFIG ===
LABEL_STUDIO_HOST = "http://localhost:8080"  # or docker IP
IMAGE_PATH = "/data/upload/6/bcec3b38-frame_1753101219502.jpg"
AUTH_TOKEN = "676c739104a47a882f6e3596b28f9efa75cf9395"  # your real token
OUTPUT = "debug_result.jpg"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# === Detections from logs ===
detections = [
    {"label": "Y", "x1": 1241.3, "y1": 713.3, "x2": 1275.2, "y2": 743.6, "conf": 0.516},
    {"label": "O", "x1": 1093.6, "y1": 909.6, "x2": 1127.6, "y2": 940.0, "conf": 0.516},
    {"label": "P", "x1": 1243.9, "y1": 123.9, "x2": 1278.9, "y2": 155.2, "conf": 0.515},
    {"label": "S", "x1": 1507.2, "y1": 303.0, "x2": 1542.7, "y2": 334.6, "conf": 0.515},
    {"label": "J", "x1": 1359.1, "y1": 501.1, "x2": 1393.4, "y2": 532.1, "conf": 0.514},
    {"label": "X", "x1": 1437.9, "y1": 824.7, "x2": 1472.9, "y2": 855.9, "conf": 0.514},
    {"label": "O", "x1": 1006.6, "y1": 613.3, "x2": 1039.7, "y2": 643.0, "conf": 0.512},
    {"label": "H", "x1": 1163.0, "y1": 276.5, "x2": 1198.5, "y2": 308.2, "conf": 0.511},
    {"label": "K", "x1": 1245.3, "y1": 429.0, "x2": 1279.9, "y2": 460.2, "conf": 0.502},
    {"label": "X", "x1": 1369.2, "y1": 213.3, "x2": 1404.3, "y2": 244.3, "conf": 0.501},
]

# === Step 1: Download Image ===
url = f"{LABEL_STUDIO_HOST.rstrip('/')}/{IMAGE_PATH.lstrip('/')}"
headers = {"Authorization": f"Token {AUTH_TOKEN}"}
response = requests.get(url, headers=headers)

if response.status_code != 200:
    raise RuntimeError(f"Failed to fetch image: {url} ({response.status_code})")

img = Image.open(io.BytesIO(response.content)).convert("RGB")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(FONT_PATH, 18)

# === Step 2: Draw Boxes ===
for det in detections:
    x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
    label = f"{det['label']} {det['conf']:.2f}"
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
    draw.text((x1, max(0, y1 - 20)), label, font=font, fill="yellow")

img.save(OUTPUT)
print(f"✅ Saved visualization: {OUTPUT}")

# ✅ Save raw (unannotated) image to root directory
raw_image_path = "debug_raw.jpg"
with open(raw_image_path, "wb") as f:
    f.write(response.content)
print(f"✅ Saved raw image: {raw_image_path}")


