from flask import Flask, request, jsonify, send_from_directory
import base64
import cv2
import numpy as np
import dlib
from imutils import face_utils
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

# Gesichts-Detektor und Landmarken-Modell laden
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Funktion Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def decode_image(base64_str):
    header, data = base64_str.split(',', 1)
    img_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.route("/api/check-fatigue", methods=["POST"])
def check_fatigue():
    data = request.get_json()
    print("Request Data:", data)

    if not data or "image" not in data:
        print("Kein Bild im Request!")
        return jsonify({"status": "kein Bild"}), 400

    image_b64 = data.get("image")
    img = decode_image(image_b64)

    if img is None:
        print("Bild konnte nicht dekodiert werden!")
        return jsonify({"status": "Bild dekodieren fehlgeschlagen"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        return jsonify({"status": "kein Gesicht erkannt"})

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Augen-Landmarken: Links 42-47, Rechts 36-41
        leftEye = shape[42:48]
        rightEye = shape[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Beispiel Schwellenwert für Müdigkeit (EAR < 0.25)
        if ear < 0.25:
            return jsonify({"status": "müde"})
        else:
            return jsonify({"status": "wach"})

    return jsonify({"status": "kein Gesicht erkannt"})

# Route für React statische Dateien
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # Falls Pfad nicht existiert, index.html ausliefern (React Routing)
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
