import os
import cv2
import base64
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# 1. Setup Folders: Ensure 'static' exists to save processed images
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 2. Load your YOLO model
# Using 'r' before the path handles Windows backslashes correctly
MODEL_PATH = r"E:\Caner detecch\New folder\runs\detect\train9\weights\best.pt"
model = YOLO(MODEL_PATH)

# ------------------ ROUTES ------------------

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

    # Save the uploaded file to the static folder
    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    file.save(input_path)

    # --- Run YOLO Inference ---
    # We use stream=True for better memory management on larger images
    results = model(input_path)

    # --- Process Results ---
    # results[0].plot() creates an image (numpy array) with boxes and labels
    annotated_img = results[0].plot()

    # Save the annotated image
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(output_path, annotated_img)

    # --- Convert to Base64 ---
    # This allows us to send the image directly to the HTML without refresh issues
    with open(output_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    return render_template(
        "index.html",
        message="Object Detection Complete!",
        output_image=f"data:image/jpeg;base64,{encoded_string}"
    )

# ------------------ RUN APP ------------------

if __name__ == "__main__":
    # debug=True allows the server to auto-reload when you save changes
    app.run(debug=True)