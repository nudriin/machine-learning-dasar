import cv2
import mediapipe as mp
import numpy as np
import time
import winsound  # untuk alarm di Windows

# --- Konfigurasi ---
CLOSED_DURATION = 3.0  # Detik mata tertutup dianggap tidur

# Mediapipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark index untuk mata (Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_idx, frame_w, frame_h):
    points = [
        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_idx
    ]
    ear = (
        euclidean_dist(points[1], points[5]) + euclidean_dist(points[2], points[4])
    ) / (2.0 * euclidean_dist(points[0], points[3]))
    return ear, points


# --- Buka webcam ---
cap = cv2.VideoCapture(0)
time.sleep(2)  # tunggu kamera siap

# --- Splash screen ---
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # mirror
    h, w, _ = frame.shape

    cv2.putText(
        frame,
        "Sleep Detection Model",
        (int(w / 6), int(h / 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3,
    )
    cv2.putText(
        frame,
        "Tekan tombol apapun untuk mulai...",
        (int(w / 8), int(h / 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Sleep Detection", frame)

    # keluar loop kalau tombol ditekan
    if cv2.waitKey(1) != -1:
        break


# --- Fungsi kalibrasi ---
def calibrate(message, duration=3):
    values = []
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # mirror
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        cv2.putText(
            frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
        )

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear, _ = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear, _ = eye_aspect_ratio(
                    face_landmarks.landmark, RIGHT_EYE, w, h
                )
                values.append((left_ear + right_ear) / 2.0)

        cv2.imshow("Sleep Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk batal
            break

    return np.mean(values) if values else 0.25


# --- Jalankan kalibrasi ---
EAR_open = calibrate("Kalibrasi: BUKA mata selama 3 detik...", 3)
EAR_closed = calibrate("Kalibrasi: TUTUP mata selama 3 detik...", 3)
EAR_THRESHOLD = (EAR_open + EAR_closed) / 2.0

print(f"Kalibrasi selesai ✅")
print(f"EAR terbuka rata-rata: {EAR_open:.3f}")
print(f"EAR tertutup rata-rata: {EAR_closed:.3f}")
print(f"Threshold EAR: {EAR_THRESHOLD:.3f}")

# --- Loop Kamera untuk deteksi real-time ---
eye_closed_time = None
alarm_on = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear, left_points = eye_aspect_ratio(
                face_landmarks.landmark, LEFT_EYE, w, h
            )
            right_ear, right_points = eye_aspect_ratio(
                face_landmarks.landmark, RIGHT_EYE, w, h
            )
            ear = (left_ear + right_ear) / 2.0

            for p in left_points + right_points:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            if ear < EAR_THRESHOLD:
                if eye_closed_time is None:
                    eye_closed_time = time.time()
                else:
                    elapsed = time.time() - eye_closed_time
                    cv2.putText(
                        frame,
                        f"Closed {elapsed:.1f}s",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    if elapsed >= CLOSED_DURATION:
                        cv2.putText(
                            frame,
                            "TERTIDUR!",
                            (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            4,
                        )
                        if not alarm_on:
                            alarm_on = True
                            winsound.Beep(1000, 2000)
            else:
                eye_closed_time = None
                alarm_on = False

    cv2.imshow("Sleep Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
