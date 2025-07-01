import cv2
import mediapipe as mp
import time
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def distance_points(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

cap = cv2.VideoCapture(0)

hand_close_start_time = None
FATIGUE_TIME_THRESHOLD = 3  # Sekunden, ab wann Müdigkeit erkannt wird
MOVEMENT_THRESHOLD = 0.01   # minimale Bewegung für "ruhig"
prev_hand_pos = None
hand_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    fatigue_detected = False

    if results_pose.pose_landmarks and results_hands.multi_hand_landmarks:
        pose_landmarks = results_pose.pose_landmarks.landmark

        # Kopf-Merkmale: Nase und Mundmitte
        nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]
        mouth_left = pose_landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = pose_landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

        mouth_center_x = (mouth_left.x + mouth_right.x) / 2
        mouth_center_y = (mouth_left.y + mouth_right.y) / 2
        mouth_pos = (mouth_center_x, mouth_center_y)
        nose_pos = (nose.x, nose.y)

        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Finger-Tipps + Handgelenk prüfen
            finger_tips_ids = [
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.WRIST
            ]

            min_dist_to_head = 1.0
            closest_tip = None
            for tip_id in finger_tips_ids:
                tip = hand_landmarks.landmark[tip_id]
                # Abstand zu Nase und Mund
                dist_to_nose = distance_points((tip.x, tip.y), nose_pos)
                dist_to_mouth = distance_points((tip.x, tip.y), mouth_pos)
                dist = min(dist_to_nose, dist_to_mouth)
                if dist < min_dist_to_head:
                    min_dist_to_head = dist
                    closest_tip = tip

            HAND_CLOSE_THRESHOLD = 0.18  # etwas größer für mehr Toleranz

            if min_dist_to_head < HAND_CLOSE_THRESHOLD:
                current_hand_pos = (closest_tip.x, closest_tip.y)

                # Bewegung berechnen
                if prev_hand_pos is not None:
                    movement = distance_points(current_hand_pos, prev_hand_pos)
                else:
                    movement = 0

                prev_hand_pos = current_hand_pos

                hand_positions.append(current_hand_pos)
                if len(hand_positions) > 10:
                    hand_positions.pop(0)

                if len(hand_positions) > 1:
                    movements = [distance_points(hand_positions[i], hand_positions[i-1]) for i in range(1, len(hand_positions))]
                    avg_movement = np.mean(movements)
                else:
                    avg_movement = 0

                # Timer läuft nur wenn Hand ruhig ist
                if avg_movement < MOVEMENT_THRESHOLD:
                    if hand_close_start_time is None:
                        hand_close_start_time = time.time()
                    else:
                        duration = time.time() - hand_close_start_time
                        if duration > FATIGUE_TIME_THRESHOLD:
                            fatigue_detected = True
                            cv2.putText(frame, "Muede: Hand am Kopf", (30, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    hand_close_start_time = None

                # Haare spielen: Hand oberhalb Nase und Bewegung vorhanden
                if closest_tip.y < nose_pos[1] and avg_movement >= MOVEMENT_THRESHOLD:
                    cv2.putText(frame, "Hand in Haarregion", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            else:
                hand_close_start_time = None
                prev_hand_pos = None
                hand_positions.clear()

    else:
        hand_close_start_time = None
        prev_hand_pos = None
        hand_positions.clear()

    if fatigue_detected:
        cv2.putText(frame, "Muedigkeit erkannt!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
