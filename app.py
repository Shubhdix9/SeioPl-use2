import os
import cv2
import base64
from flask import Flask, render_template, request
from ultralytics import YOLO
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
MODEL_PATH = r"E:\Caner detecch\New folder\runs\detect\train9\weights\best.pt"
model = YOLO(MODEL_PATH)
@app.route("/")
def home():
    """Renders the main upload page."""
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, runs YOLO, and returns the result."""
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

    app.run()
