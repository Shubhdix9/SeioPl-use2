import os
import cv2
import base64
from flask import Flask, render_template, request
from ultralytics import YOLO
import gdown  # make sure gdown is in requirements.txt

app = Flask(__name__)

# ------------------ FOLDER SETUP ------------------
UPLOAD_FOLDER = 'static'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ------------------ YOLO MODEL ------------------
MODEL_PATH = os.path.join(MODEL_FOLDER, "best.pt")

# Download model from Google Drive if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    gdrive_url = "https://drive.google.com/uc?id=1SXWo95eJmeHjXEEsoBJmWQjrHvcrwkfG"
    gdown.download(gdrive_url, MODEL_PATH, quiet=False)

# Load YOLO model
model = YOLO(MODEL_PATH)

# ------------------ ROUTES ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400
    file = request.files["image"]
    if file.filename == '':
        return "No selected file", 400

    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    file.save(input_path)

    results = model(input_path)
    annotated_img = results[0].plot()

    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(output_path, annotated_img)

    with open(output_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    return render_template(
        "index.html",
        message="Object Detection Complete!",
        output_image=f"data:image/jpeg;base64,{encoded_string}"
    )

if __name__ == "__main__":
    app.run(debug=True)
