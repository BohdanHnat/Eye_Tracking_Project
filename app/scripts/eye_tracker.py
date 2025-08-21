from app.scripts.settings import LEFT_IRIS, RIGHT_IRIS, ET_RECORD_FRAMES
from app.scripts.calculations import get_iris_center, normalize_iris
from app.utils import frame_data_lock
import cv2

def run_eye_tracking(cam_frame, frame_idx, face_mesh, mean_center, session_id, gaze_store, image_filename):
    if frame_idx < ET_RECORD_FRAMES:
        rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = cam_frame.shape

        # LEFT EYE
        left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
        left_inner = [landmarks[133].x * w, landmarks[133].y * h]
        left_outer = [landmarks[33].x * w, landmarks[33].y * h]
        norm_left = normalize_iris(left_iris, left_inner, left_outer)

        # RIGHT EYE
        right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
        right_inner = [landmarks[263].x * w, landmarks[263].y * h]
        right_outer = [landmarks[362].x * w, landmarks[362].y * h]
        norm_right = normalize_iris(right_iris, right_inner, right_outer)

        # Average both eyes
        norm_iris = [
            (norm_left[0] + norm_right[0]) / 2,
            (norm_left[1] + norm_right[1]) / 2
        ]

        # Center normalized iris value using calibration mean
        norm_iris_centered = [
            norm_iris[0] - mean_center[0],
            norm_iris[1] - mean_center[1]
        ]

        # Append to a per-image list in gaze_store
        # Structure: gaze_store[session_id]['gaze_results'][image_filename] -> [{'frame', 'norm_x', 'norm_y'}, {...}]
        session_store = gaze_store.setdefault(session_id, {})
        gaze_results_dict = session_store.setdefault('gaze_results', {})
        frames_list = gaze_results_dict.setdefault(image_filename, [])

        # Use existing lock to avoid races across threads
        with frame_data_lock:
            frames_list.append({
                'frame': frame_idx,
                'norm_x': norm_iris_centered[0],
                'norm_y': norm_iris_centered[1]
            })


