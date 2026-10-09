from datetime import datetime
from math import hypot
from pathlib import Path
import ctypes
from collections import deque

import cv2
import mediapipe as mp


GESTURE_ACTIONS = {
    "Open Palm": "Play / Pause",
    "Fist": "Stop",
    "Thumb Up": "Increase volume",
    "Thumb Down": "Decrease volume",
    "Victory Sign": "Take screenshot",
}

MEDIA_KEYS = {
    "Open Palm": 0xB3,
    "Fist": 0xB2,
    "Thumb Up": 0xAF,
    "Thumb Down": 0xAE,
}


def classify_gesture(hand_landmarks):
    points = hand_landmarks.landmark
    finger_extended = [
        points[tip].y < points[pip].y
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    ]
    thumb_extended = hypot(
        points[4].x - points[5].x, points[4].y - points[5].y
    ) > hypot(points[3].x - points[5].x, points[3].y - points[5].y) * 1.15

    if all(finger_extended) and thumb_extended:
        return "Open Palm"
    if not any(finger_extended) and not thumb_extended:
        return "Fist"
    if finger_extended[:2] == [True, True] and not any(finger_extended[2:]):
        return "Victory Sign"
    if not any(finger_extended) and thumb_extended:
        if points[4].y < points[2].y - 0.04:
            return "Thumb Up"
        if points[4].y > points[2].y + 0.04:
            return "Thumb Down"
    return None


def perform_action(gesture):
    if gesture == "Victory Sign":
        return GESTURE_ACTIONS[gesture]

    key = MEDIA_KEYS.get(gesture)
    if key is None:
        return GESTURE_ACTIONS[gesture]

    user32 = ctypes.windll.user32
    user32.keybd_event(key, 0, 0, 0)
    user32.keybd_event(key, 0, 0x0002, 0)
    return GESTURE_ACTIONS[gesture]


def main():
    try:
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
    except AttributeError as exc:
        raise RuntimeError(
            "This script requires MediaPipe 0.10.21. Run it with the project's "
            ".venv-gesture Python interpreter."
        ) from exc

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check that it is connected and available.")

    screenshot_dir = Path(__file__).resolve().parent / "screenshots"
    candidate_gesture = None
    candidate_frames = 0
    active_gesture = None
    last_action = "Show a gesture"
    action_history = deque(maxlen=4)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Could not read a frame from the webcam.")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                gesture = None

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    gesture = classify_gesture(hand_landmarks)

                if gesture is None:
                    candidate_gesture = None
                    candidate_frames = 0
                    active_gesture = None
                elif gesture == candidate_gesture:
                    candidate_frames += 1
                else:
                    candidate_gesture = gesture
                    candidate_frames = 1

                screenshot_requested = False
                action_triggered = False
                if candidate_frames >= 6 and candidate_gesture != active_gesture:
                    active_gesture = candidate_gesture
                    action_triggered = True
                    screenshot_requested = active_gesture == "Victory Sign"
                    last_action = perform_action(active_gesture)

                label = gesture or "No recognized gesture"
                action = GESTURE_ACTIONS.get(gesture, "—")
                cv2.putText(
                    frame,
                    f"Gesture: {label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Action: {action}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Last: {last_action}",
                    (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                if screenshot_requested:
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = screenshot_dir / (
                        f"screenshot_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                    )
                    if not cv2.imwrite(str(screenshot_path), frame):
                        raise OSError(f"Could not save screenshot to {screenshot_path}")
                    last_action = f"Screenshot saved: {screenshot_path.name}"

                if action_triggered:
                    if screenshot_requested:
                        action_history.append(
                            f"{datetime.now():%H:%M:%S} - Victory Sign: Screenshot saved"
                        )
                    elif candidate_gesture:
                        action_history.append(
                            f"{datetime.now():%H:%M:%S} - "
                            f"{candidate_gesture}: {last_action}"
                        )

                history_top = frame.shape[0] - 18 * (len(action_history) + 1) - 12
                panel_width = min(frame.shape[1] - 20, 430)
                cv2.rectangle(
                    frame,
                    (10, history_top),
                    (10 + panel_width, frame.shape[0] - 10),
                    (35, 35, 35),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    "Recent actions:",
                    (18, history_top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 220, 255),
                    1,
                )
                for index, entry in enumerate(reversed(action_history), start=1):
                    cv2.putText(
                        frame,
                        entry,
                        (18, history_top + 20 * (index + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow("Hand Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
