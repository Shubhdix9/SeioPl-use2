import os
import cv2
import base64
from flask import Flask, render_template, request
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

# ------------------ APP SETUP ------------------
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ------------------ MODEL SETUP ------------------
MODEL_PATH = os.path.join(MODEL_FOLDER, "best.pt")

# Safe global to prevent torch unpickling errors
torch.serialization.safe_globals([DetectionModel])

# Load YOLO model once
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ------------------ ROUTES ------------------
@app.route("/")
def home():
    """Render the main upload page."""
    return render_template("index.html", output_image=None, message=None)

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, run YOLO inference, and return result."""
    if model is None:
        return "Model is not loaded", 500

    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    file.save(input_path)

    # Run YOLO inference
    try:
        results = model(input_path)
        annotated_img = results[0].plot()
    except Exception as e:
        return f"Error during inference: {e}", 500

    # Save output image
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(output_path, annotated_img)

    # Encode image as Base64 for HTML embedding
    with open(output_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    return render_template(
        "index.html",
        output_image=f"data:image/jpeg;base64,{encoded_string}",
        message="Detection Complete!"
    )

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT automatically
    app.run(host="0.0.0.0", port=port, debug=False)
