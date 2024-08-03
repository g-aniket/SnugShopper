import traceback
import cv2
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin

from bodyDetection import get_body_measurements
from main import get_measurements_from_image

app = Flask(__name__)
cors = CORS(app, origins="*")

VERSION = "v-0.3.0"


@app.route("/")
@cross_origin()
def hello_world():
    return "Hello World! " + VERSION


@app.route("/get-measurements", methods=["POST"])
@cross_origin()
def process_image():
    try:
        if "image" not in request.files:
            return str({"error": "No image file found in request"}), 401

        image_file = request.files["image"]
        if not image_file.mimetype.startswith("image/"):
            return str({"error": "Invalid image format"}), 402

        img_string = np.fromfile(image_file, np.uint8)
        image = cv2.imdecode(img_string, cv2.IMREAD_COLOR)

        measurements = get_measurements_from_image(image, True)
        if "error" in measurements:
            return str({"error": measurements["message"]}), 450

        res = {}
        for key, value in measurements.items():
            res[key] = round(value, 2)
        return str(res), 200

    except Exception as e:
        return str({"error": str(e)}), 500


@app.route("/get-body-measurements", methods=["POST"])
@cross_origin()
def process_body_image():
    try:
        if (
            "height" not in request.form
            or "front" not in request.files
            or "side" not in request.files
        ):
            return str({"error": "Missing required parameters"}), 401

        height = float(request.form["height"])
        front_img = request.files["front"]
        side_img = request.files["side"]

        if not front_img.mimetype.startswith("image/"):
            return str({"error": "Invalid front image format"}), 402
        if not side_img.mimetype.startswith("image/"):
            return str({"error": "Invalid side image format"}), 402

        front_string = np.fromfile(front_img, np.uint8)
        side_string = np.fromfile(side_img, np.uint8)

        front = cv2.imdecode(front_string, cv2.IMREAD_COLOR)
        side = cv2.imdecode(side_string, cv2.IMREAD_COLOR)

        measurements = get_body_measurements(height, front, side)
        if "error" in measurements:
            return str({"error": measurements["error"]}), 450

        return str(measurements), 200

    except Exception as e:
        traceback.print_exc()
        return str({"error": str(e)}), 500
