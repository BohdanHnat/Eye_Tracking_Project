from app.scripts.visuals import draw_target
from app.scripts.calculations import get_iris_center, normalize_iris
from app.scripts.settings import LEFT_IRIS, RIGHT_IRIS, TARGET_POSITIONS, CALIB_FRAMES
import numpy as np
import cv2

def gen_calibration_frames(face_mesh, camera, calib_started, target_index, target_frames, session_id, gaze_store):
    if session_id in gaze_store and 'gaze_values' in gaze_store[session_id]:
        gaze_values = gaze_store[session_id]['gaze_values']
    else:
        gaze_values = {pos.lower(): [] for pos in TARGET_POSITIONS}

        if session_id not in gaze_store:
            gaze_store[session_id] = {}

        gaze_store[session_id]['gaze_values'] = gaze_values

    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        frame_disp = frame.copy()

        if not calib_started:
            cv2.putText(
                frame_disp, "Press 'Start Calibration' to begin",
                (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA
            )
        else:
            idx = target_index
            count = target_frames

            if idx < len(TARGET_POSITIONS):
                pos = TARGET_POSITIONS[idx]
                frame_disp = draw_target(frame_disp, target_position=pos.lower())
                cv2.putText(
                    frame_disp, f"{pos.replace('_', ' ').title()} ({count + 1}/{CALIB_FRAMES})",
                    (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA
                )

                # Run landmark detection and collect gaze point
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    h, w, _ = frame.shape

                    # Left and Right eye calculations
                    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
                    left_inner = [landmarks[133].x * w, landmarks[133].y * h]
                    left_outer = [landmarks[33].x * w, landmarks[33].y * h]
                    norm_left = normalize_iris(left_iris, left_inner, left_outer)

                    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                    right_inner = [landmarks[263].x * w, landmarks[263].y * h]
                    right_outer = [landmarks[362].x * w, landmarks[362].y * h]
                    norm_right = normalize_iris(right_iris, right_inner, right_outer)

                    norm_iris = [
                        (norm_left[0] + norm_right[0]) / 2,
                        (norm_left[1] + norm_right[1]) / 2
                    ]
                    # Collect data for this target only if not finished
                    if count < CALIB_FRAMES:
                        gaze_values[pos.lower()].append(norm_iris)
                        # Persist update for other generator calls
                        gaze_store[session_id]['gaze_values'] = gaze_values
                        target_frames += 1
            else:
                # Calibration complete

                cv2.putText(
                    frame_disp, "Calibration complete!", (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 0), 4, cv2.LINE_AA
                )

                # --- Post-process calibration data ---
                calibration_results = {}
                for pos in TARGET_POSITIONS:
                    values = gaze_values[pos.lower()]
                    if len(values) > 0:
                        mean = np.mean(values, axis=0)
                    else:
                        mean = (0.0, 0.0)
                    calibration_results[pos.lower()] = (mean[0], mean[1])

                all_means = np.array(list(calibration_results.values()))
                mean_x = np.mean(all_means[:, 0])
                mean_y = np.mean(all_means[:, 1])

                # Center all means around (0,0)
                for key in calibration_results:
                    x, y = calibration_results[key]
                    calibration_results[key] = (x - mean_x, y - mean_y)

                # Save all results into the gaze_store for this session
                gaze_store[session_id].update({
                    'calib_results': calibration_results,
                    'mean_center': (mean_x, mean_y),
                    'raw_values': gaze_values  # (Optional: remove if you don't want to keep all raw values)
                })

        ret, buffer = cv2.imencode('.jpg', frame_disp)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
