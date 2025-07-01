from flask import Flask, request, jsonify, send_from_directory
import base64
import cv2
import numpy as np
import dlib
from imutils import face_utils
from flask_cors import CORS
import os
import mediapipe as mp  # MediaPipe importieren

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
hog_detector = dlib.get_frontal_face_detector()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(shape, img_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    image_points = np.array([
        shape[30],
        shape[8],
        shape[36],
        shape[45],
        shape[48],
        shape[54]
    ], dtype="double")

    size = (img_shape[0], img_shape[1])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()

    return pitch, yaw, roll

def get_head_pose_mediapipe(landmarks, img_shape):
    # Modellpunkte (3D) ungefähr entsprechend dlib-Punkte (in mm)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nasenwurzel
        (0.0, -330.0, -65.0),        # Kinn
        (-225.0, 170.0, -135.0),     # linkes Auge links außen
        (225.0, 170.0, -135.0),      # rechtes Auge rechts außen
        (-150.0, -150.0, -125.0),    # linker Mundwinkel
        (150.0, -150.0, -125.0)      # rechter Mundwinkel
    ])

    # MediaPipe Landmark Indices für diese Punkte (2D Bildpunkte)
    image_points = np.array([
        landmarks[1],     # Nasenwurzel (index 1)
        landmarks[152],   # Kinn (index 152)
        landmarks[33],    # linkes Auge außen (index 33)
        landmarks[263],   # rechtes Auge außen (index 263)
        landmarks[78],    # linker Mundwinkel (index 78)
        landmarks[308],   # rechter Mundwinkel (index 308)
    ], dtype="double")

    size = (img_shape[0], img_shape[1])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # keine Verzerrung

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll


def decode_image(base64_str):
    try:
        header, data = base64_str.split(',', 1)
        missing_padding = len(data) % 4
        if missing_padding:
            data += '=' * (4 - missing_padding)
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode gibt None zurück")
        return img
    except Exception as e:
        print("Fehler beim Bild-Decodieren:", e)
        return None

def gaze_direction(eye_points, gray_img):
    margin = 5
    min_x = np.min(eye_points[:, 0]) - margin
    max_x = np.max(eye_points[:, 0]) + margin
    min_y = np.min(eye_points[:, 1]) - margin
    max_y = np.max(eye_points[:, 1]) + margin

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, gray_img.shape[1])
    max_y = min(max_y, gray_img.shape[0])

    eye_roi = gray_img[min_y:max_y, min_x:max_x]

    if eye_roi.size == 0:
        return "unknown"

    _, thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"

    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return "unknown"
    cx = int(M["m10"] / M["m00"])

    eye_width = max_x - min_x
    gaze_ratio = cx / eye_width

    if gaze_ratio < 0.4:
        return "left"
    elif gaze_ratio > 0.6:
        return "right"
    else:
        return "center"

def lernfaehigkeits_score(ear, left_gaze, right_gaze, pitch, yaw):
    score = 0

    if ear >= 0.25:
        score += 40
    else:
        score += 0

    if left_gaze == "center" and right_gaze == "center":
        score += 30
    elif left_gaze == "center" or right_gaze == "center":
        score += 20
    else:
        score += 10

    if abs(yaw) <= 15 and abs(pitch) <= 15:
        score += 30
    else:
        score += 10

    return score

def get_landmarks_mediapipe(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        h, w, _ = image.shape
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))
        return np.array(landmarks)
    return None


def resize_image_keep_aspect(img, max_side=640):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

@app.route("/api/check-fatigue", methods=["POST"])
def check_fatigue():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"status": "kein Bild"}), 400

    image_b64 = data.get("image")
    img = decode_image(image_b64)
    if img is None:
        return jsonify({"status": "Bild dekodieren fehlgeschlagen"}), 400

    img = resize_image_keep_aspect(img, 640)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = hog_detector(gray, 1)

    if len(rects) > 0:
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        method_used = "dlib"
    else:
        # Kein Gesicht mit dlib, MediaPipe versuchen
        shape = get_landmarks_mediapipe(img)
        method_used = "mediapipe"
        if shape is None:
            return jsonify({"status": "kein Gesicht erkannt"})

    # Falls MediaPipe: keine 68 Punkte wie dlib, sondern 468. Für Augen + Head Pose nehmen wir Schlüsselindizes (Anpassung nötig)

    if method_used == "dlib":
        leftEye = shape[36:42]
        rightEye = shape[42:48]
    else:
        # MediaPipe: Augenindizes grob approximieren
        leftEye_indices = [33, 160, 158, 133, 153, 144]  # Beispiel für linkes Auge
        rightEye_indices = [263, 387, 385, 362, 380, 373]  # Beispiel für rechtes Auge
        leftEye = np.array([shape[i] for i in leftEye_indices])
        rightEye = np.array([shape[i] for i in rightEye_indices])

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    ear_percent = round(ear * 100, 1)

    status_fatigue = "tired" if ear < 0.25 else "awake"

    left_gaze = gaze_direction(leftEye, gray)
    right_gaze = gaze_direction(rightEye, gray)

    # Head pose nur bei dlib möglich, bei MediaPipe hier eine Dummy-Ausgabe (kann man später mit mp.solutions.face_mesh erweitern)
    if method_used == "dlib":
        pitch, yaw, roll = get_head_pose(shape, img.shape)
    else:
        pitch, yaw, roll = get_head_pose_mediapipe(shape, img.shape)

    attention_head_pose = "aufmerksam"
    if abs(yaw) > 20 or abs(pitch) > 15:
        attention_head_pose = "abgelenkt"

    attention_count = 0
    if ear >= 0.25:
        attention_count += 1
    if left_gaze == "center" and right_gaze == "center":
        attention_count += 1
    if attention_head_pose == "aufmerksam":
        attention_count += 1

    attention_status = "aufmerksam" if attention_count >= 2 else "abgelenkt"

    lernfaehigkeits_score_value = lernfaehigkeits_score(ear, left_gaze, right_gaze, pitch, yaw)

    print(f"Method: {method_used}, EAR: {ear_percent}, Gaze: {left_gaze}/{right_gaze}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")
    print(f"Attention count: {attention_count}, Status: {attention_status}, Lernfähigkeits-Score: {lernfaehigkeits_score_value}")

    return jsonify({
        "methodUsed": method_used,
        "status": status_fatigue,
        "avg EAR": ear_percent,
        "facesDetected": len(rects),
        "gazeLeft": left_gaze,
        "gazeRight": right_gaze,
        "headPose": {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        },
        "attention": attention_status,
        "lernfaehigkeitsScore": lernfaehigkeits_score_value
    })

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
