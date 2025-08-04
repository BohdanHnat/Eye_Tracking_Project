from app.scripts.calculations import get_iris_center, normalize_iris
from app.scripts.settings import LEFT_IRIS, RIGHT_IRIS, ET_RECORD_FRAMES
import cv2

def run_eye_tracking(face_mesh, camera, mean_center, session_id, gaze_store, image_filename):
    gaze_data = []
    frame_idx = 0

    # Core tracking loop
    while camera.isOpened() and frame_idx < ET_RECORD_FRAMES:
        success, frame = camera.read()
        if not success:
            continue

        frame_disp = frame.copy()
        cv2.putText(
            frame_disp, f"Record Frames: ({frame_idx + 1}/{ET_RECORD_FRAMES})",
            (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # --- LEFT EYE ---
        left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
        left_inner = [landmarks[133].x * w, landmarks[133].y * h]
        left_outer = [landmarks[33].x * w, landmarks[33].y * h]
        norm_left = normalize_iris(left_iris, left_inner, left_outer)

        # --- RIGHT EYE ---
        right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
        right_inner = [landmarks[263].x * w, landmarks[263].y * h]
        right_outer = [landmarks[362].x * w, landmarks[362].y * h]
        norm_right = normalize_iris(right_iris, right_inner, right_outer)

        # --- Average both eyes ---
        norm_iris = [
            (norm_left[0] + norm_right[0]) / 2,
            (norm_left[1] + norm_right[1]) / 2
        ]

        # --- Center normalized iris value using calibration mean ---
        norm_iris_centered = [
            norm_iris[0] - mean_center[0],
            norm_iris[1] - mean_center[1]
        ]

        # --- Save data ---
        gaze_data.append({
            'frame': frame_idx,
            'norm_x': norm_iris_centered[0],
            'norm_y': norm_iris_centered[1]
        })

        frame_idx += 1

    # Save to gaze_store under the image filename key, in a 'results' dict
    session_store = gaze_store.setdefault(session_id, {})
    gaze_results_dict = session_store.setdefault('gaze_results', {})
    gaze_results_dict[image_filename] = gaze_data


