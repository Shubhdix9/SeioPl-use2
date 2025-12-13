import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO
import gdown

app = Flask(__name__)

# ------------------ FOLDER SETUP ------------------
UPLOAD_FOLDER = "static"
MODEL_FOLDER = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ------------------ YOLO MODEL ------------------
MODEL_PATH = os.path.join(MODEL_FOLDER, "best.pt")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    gdrive_url = "https://drive.google.com/uc?id=1SXWo95eJmeHjXEEsoBJmWQjrHvcrwkfG"
    gdown.download(gdrive_url, MODEL_PATH, quiet=False)

# Load model (CPU only)
model = YOLO(MODEL_PATH)
model.to("cpu")

# ------------------ ROUTES ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # ✅ MUST MATCH HTML name="file"
    file = request.files.get("file")

    if not file or file.filename == "":
        return "No file uploaded", 400

    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")

    # Save uploaded image
    file.save(input_path)

    # Run YOLO
    results = model(input_path)
    annotated_img = results[0].plot()

    # Save output image
    cv2.imwrite(output_path, annotated_img)

    return render_template(
        "index.html",
        output_image="/static/output.jpg"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
