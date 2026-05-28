"""
Live camera demo: open the webcam, run MediaPipe FaceMesh, and draw the
detected face landmarks in real time. Also shows the live EAR / MAR values.

This is just a visualization sanity-check for the MediaPipe pipeline -
it does NOT run the LSTM classifier (that model is not trained yet).

Controls:
    q / ESC  - quit
    m        - toggle full face mesh on/off (keeps the EAR/MAR points)

Run:
    python live_demo.py
"""
import cv2
import numpy as np
import mediapipe as mp

RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 82, 312, 308, 317, 87]


def _coords(landmarks, indices, w, h):
    return np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                     for i in indices])


def _aspect_ratio(p):
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal == 0:
        return 0.0
    return (v1 + v2) / (2.0 * horizontal)


def main(camera_index=0):
    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {camera_index}")

    show_mesh = True
    print("Camera opened. Press 'q' or ESC to quit, 'm' to toggle the mesh.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels natural
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_text, mar_text = "EAR: --", "MAR: --"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            if show_mesh:
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                    .get_default_face_mesh_tesselation_style(),
                )

            # Highlight the exact points used for EAR / MAR in green.
            for idx in RIGHT_EYE_INDICES + LEFT_EYE_INDICES + MOUTH_INDICES:
                lm = face_landmarks.landmark[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2,
                           (0, 255, 0), -1)

            right_eye = _coords(face_landmarks, RIGHT_EYE_INDICES, w, h)
            left_eye = _coords(face_landmarks, LEFT_EYE_INDICES, w, h)
            mouth = _coords(face_landmarks, MOUTH_INDICES, w, h)

            ear = (_aspect_ratio(right_eye) + _aspect_ratio(left_eye)) / 2.0
            mar = _aspect_ratio(mouth)
            ear_text, mar_text = f"EAR: {ear:.3f}", f"MAR: {mar:.3f}"
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, ear_text, (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, mar_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("FaceMesh live demo (q=quit, m=toggle mesh)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("m"):
            show_mesh = not show_mesh

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
