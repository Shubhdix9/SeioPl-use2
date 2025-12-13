import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = "static"
MODEL_FOLDER = "models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

model = None  # 👈 GLOBAL BUT EMPTY

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading YOLO model...")
            gdown.download(
                "https://drive.google.com/uc?id=1SXWo95eJmeHjXEEsoBJmWQjrHvcrwkfG",
                MODEL_PATH,
                quiet=False
            )
        model = YOLO(MODEL_PATH)
        model.to("cpu")
    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return "No file uploaded", 400

    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    file.save(input_path)

    model = load_model()  # 👈 LOAD ONLY WHEN NEEDED
    results = model(input_path)
    annotated = results[0].plot()
    cv2.imwrite(output_path, annotated)

    return render_template(
        "index.html",
        output_image="/static/output.jpg"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
