"""
Feature extraction: turn a video into a (num_frames, 2) array of [EAR, MAR].

EAR = Eye Aspect Ratio  -> drops when eyes close
MAR = Mouth Aspect Ratio -> rises when the mouth opens (yawning / talking)

This logic is the same maths as the original main.py, refactored so both the
preprocessing script and a real-time demo can reuse it.
"""
import cv2
import numpy as np
import mediapipe as mp

from config import RIGHT_EYE_INDICES, LEFT_EYE_INDICES, MOUTH_INDICES


def _get_coords(landmarks, indices, img_w, img_h):
    return np.array(
        [[landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h]
         for i in indices]
    )


def _aspect_ratio(points):
    """Generic 6-point aspect ratio: (|p1-p5| + |p2-p4|) / (2*|p0-p3|)."""
    v1 = np.linalg.norm(points[1] - points[5])
    v2 = np.linalg.norm(points[2] - points[4])
    h = np.linalg.norm(points[0] - points[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_ear_mar(face_landmarks, img_w, img_h):
    """Return (ear, mar) for a single detected face."""
    right_eye = _get_coords(face_landmarks, RIGHT_EYE_INDICES, img_w, img_h)
    left_eye = _get_coords(face_landmarks, LEFT_EYE_INDICES, img_w, img_h)
    mouth = _get_coords(face_landmarks, MOUTH_INDICES, img_w, img_h)

    ear = (_aspect_ratio(right_eye) + _aspect_ratio(left_eye)) / 2.0
    mar = _aspect_ratio(mouth)
    return ear, mar


def extract_features_from_video(video_path, verbose=False):
    """
    Process one video file and return a float32 array of shape (num_frames, 2).

    Frames where no face is detected reuse the previous valid value
    (forward-fill), or zeros if no face has been seen yet. This avoids the
    sharp zero-spikes that the original script produced on missed detections.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        face_mesh.close()
        raise IOError(f"Could not open video: {video_path}")

    features = []
    last_valid = [0.0, 0.0]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            ear, mar = compute_ear_mar(results.multi_face_landmarks[0],
                                       img_w, img_h)
            last_valid = [ear, mar]

        features.append(list(last_valid))
        frame_count += 1
        if verbose and frame_count % 50 == 0:
            print(f"    processed {frame_count} frames...")

    cap.release()
    face_mesh.close()
    return np.asarray(features, dtype=np.float32)
