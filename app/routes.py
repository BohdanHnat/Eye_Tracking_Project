from flask import render_template, redirect, url_for, session, Response, send_file, current_app, request

from app import app
from app.scripts.calibration import gen_calibration_frames
from app.scripts.eye_tracker import run_eye_tracking
from app.scripts.settings import face_mesh, camera, TARGET_POSITIONS, ET_RECORD_FRAMES, CAM_FPS
from app.scripts.visuals import generate_heatmap
from app.utils import get_session_id, gaze_store, get_image_files

import threading
import io
import os

@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session.clear()
    # Clear gaze_store for this session
    sid = get_session_id()
    if sid in gaze_store:
        del gaze_store[sid]
    return redirect(url_for('main_page'))

# Displays calibration page (video feed and buttons).
# Ensures 'calib_started' flag exists in session, defaulting to False
@app.route('/calibration', methods=['GET', 'POST'])
def calibration():

    if 'calib_started' not in session:
        session['calib_started'] = False

    num_targets = len(TARGET_POSITIONS)

    calib_completed = session.get('target_index', 0) >= num_targets

    return render_template('calibration.html', calib_completed=calib_completed)

# Real-time video from the webcam - gaze detection.
# Live feed to be displayed in the <img> tag on the calibration page.
@app.route('/video_feed')
def video_feed():
    # Current calibration progress
    calib_started = session.get('calib_started', False)
    target_index = session.get('target_index', 0) # Get current target
    target_frames = session.get('target_frames', 0) # Get frames count so far

    sid = get_session_id()

    return Response(
        gen_calibration_frames(face_mesh, camera, calib_started, target_index, target_frames, sid, gaze_store),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_calibration', methods=['POST'])
def start_calibration():
    # Clear all previous session data ONLY when (re)starting calibration
    session.clear()
    session['calib_started'] = True
    session['target_index'] = 0
    session['target_frames'] = 0

    # Optionally clear gaze_store as well for this session
    sid = get_session_id()
    if sid in gaze_store:
        del gaze_store[sid]

    return redirect(url_for('calibration'))

@app.route('/next_target', methods=['POST'])
def next_target():
    # Only allow advancing targets if calibration has started
    if not session.get('calib_started', False):
        return redirect(url_for('calibration'))

    # Move to next calibration target and reset frame counter (for UI/generator state)
    idx = session.get('target_index', 0)
    session['target_index'] = idx + 1
    session['target_frames'] = 0

    return redirect(url_for('calibration'))

@app.route('/calibration_results')
def calibration_results():
    sid = get_session_id()
    gaze_store_data = gaze_store.get(sid, {})

    # Defensive: If no results yet, redirect to calibration
    if 'calib_results' not in gaze_store_data or 'mean_center' not in gaze_store_data:
        return redirect(url_for('calibration'))

    return render_template(
        'calib_results.html',
        calib_results=gaze_store_data['calib_results'],
        mean_center=gaze_store_data['mean_center'],
        raw_values=gaze_store_data.get('raw_values', {})
    )

@app.route('/eye_tracking', methods=['GET'])
def eye_tracking():
    sid = get_session_id()
    image_files = get_image_files()

    selected_image = request.args.get('image', image_files[0] if image_files else 'img_1.jpg')

    tracking_in_progress = gaze_store.get(sid, {}).get('tracking_in_progress', False)
    tracking_seconds = ET_RECORD_FRAMES // CAM_FPS

    return render_template(
        'eye_tracking.html',
        image_files=image_files,
        selected_image=selected_image,
        tracking_in_progress=tracking_in_progress,
        tracking_seconds=tracking_seconds
    )

@app.route('/start_eye_tracking')
def start_eye_tracking():
    sid = get_session_id()
    image = request.args.get('image')

    # Mark tracking as in progress
    gaze_store.setdefault(sid, {})['tracking_in_progress'] = True

    def run_tracking():
        mean_center = gaze_store[sid].get('mean_center', (0.0, 0.0))

        run_eye_tracking(face_mesh, camera, mean_center, sid, gaze_store, image)

        gaze_store[sid]['tracking_in_progress'] = False

    # Start background thread
    threading.Thread(target=run_tracking).start()

    # Redirect to main eye tracking page (will show notification)
    return redirect(url_for('eye_tracking', image=image))

@app.route('/eye_tracking_status')
def eye_tracking_status():
    sid = get_session_id()
    status = gaze_store.get(sid, {}).get('tracking_in_progress', False)
    return {'in_progress': status}

@app.route('/eye_tracking_results')
def eye_tracking_results():
    sid = get_session_id()

    # Get dictionary of all image gaze results for this session
    gaze_results_dict = gaze_store.get(sid, {}).get('gaze_results', {})  # {image_filename: gaze_data, ...}
    processed_images = list(gaze_results_dict.keys())

    selected_image = request.args.get('image')

    if not selected_image and processed_images:
        selected_image = processed_images[-1]
    elif not processed_images:
        selected_image = None  # Nothing tracked yet

    return render_template(
        'eye_tracking_results.html',
        processed_images=processed_images,
        selected_image=selected_image,
    )

@app.route('/gaze_heatmap')
def gaze_heatmap():
    sid = get_session_id()
    image = request.args.get('image')

    # Look up the gaze data for the selected image
    gaze_data = gaze_store.get(sid, {}).get('gaze_results', {}).get(image, [])

    bg_image_file = os.path.join(current_app.static_folder, image)

    img = generate_heatmap(gaze_data, bg_image_file)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')
