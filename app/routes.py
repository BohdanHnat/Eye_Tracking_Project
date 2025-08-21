from flask import render_template, redirect, url_for, session, send_file, current_app, request, jsonify

from app.utils import get_session_id, get_image_files, gaze_store, frame_counters, frame_data_lock
from app.scripts.settings import face_mesh, TARGET_POSITIONS, ET_RECORD_FRAMES, CAM_FPS
from app.scripts.eye_tracker import run_eye_tracking
from app.scripts.calibration import calibrate_frame
from app.scripts.visuals import generate_heatmap
from app import app

import numpy as np
import threading
import base64
import cv2
import io
import os

@app.route('/')
def homepage():
    session['calib_started'] = False
    return render_template('homepage.html')

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session.clear()

    # Clear gaze_store for this session
    sid = get_session_id()
    if sid in gaze_store:
        del gaze_store[sid]

    return redirect(url_for('homepage'))

@app.route('/calibration', methods=['GET', 'POST'])
def calibration():
    if 'calib_started' not in session:
        session['calib_started'] = False

    num_targets = len(TARGET_POSITIONS)
    calib_completed = session.get('target_index', 0) >= num_targets

    return render_template('calibration.html', calib_completed=calib_completed)

@app.route('/calibration_frame', methods=['POST'])
def calibration_frame(): # Recieves a webcam frame
    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Fetch session and calibration state
    session_id = get_session_id()
    calib_started = session.get('calib_started', False)
    target_index = session.get('target_index', 0)
    current_target_frame = session.get('current_target_frame', 0)

    frame_bytes, current_target_frame = calibrate_frame(
        frame, face_mesh, calib_started, target_index, current_target_frame, session_id, gaze_store
    )

    session['current_target_frame'] = current_target_frame

    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

    return jsonify({'frame': frame_base64})

@app.route('/start_calibration', methods=['POST'])
def start_calibration():
    # Clear all previous session data upon (re)starting calibration
    session.clear()
    session['calib_started'] = True
    session['target_index'] = 0
    session['current_target_frame'] = 0

    # Clear gaze_store as well for this session
    sid = get_session_id()
    if sid in gaze_store:
        del gaze_store[sid]

    return redirect(url_for('calibration'))

@app.route('/next_target', methods=['POST'])
def next_target():
    # Allow advancing targets only if calibration has started
    if not session.get('calib_started', False):
        return redirect(url_for('calibration'))

    # Move to next calibration target and reset frame counter
    idx = session.get('target_index', 0)
    session['target_index'] = idx + 1
    session['current_target_frame'] = 0

    return redirect(url_for('calibration'))

@app.route('/eye_tracking', methods=['GET'])
def eye_tracking():
    sid = get_session_id()
    image_files = get_image_files()
    session['calib_started'] = False

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

@app.route('/eye_tracking_frame', methods=['POST'])
def eye_tracking_frame(): # Receives a webcam frame and starts a background thread to process it
    sid = get_session_id()
    image_name = request.args.get('image')
    frame_file = request.files.get('frame')

    if not image_name or not frame_file:
        return jsonify({'error': 'Missing image name or frame file'}), 400

    # Read and decode the frame data immediately
    frame_bytes = frame_file.read()
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame_for_tracking = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Read the data into a NumPy array

    with frame_data_lock:
        current_et_frame = frame_counters.get(sid, 0)
        frame_counters[sid] = current_et_frame + 1

    def run_tracking_in_thread():
        mean_center = gaze_store.get(sid, {}).get('mean_center', (0.0, 0.0))

        run_eye_tracking(frame_for_tracking, current_et_frame, face_mesh, mean_center, sid, gaze_store, image_name)

    threading.Thread(target=run_tracking_in_thread).start()

    return jsonify({'status': 'Frame accepted for processing'}), 202

@app.route('/start_eye_tracking')
def start_eye_tracking():
    sid = get_session_id()
    image = request.args.get('image')

    gaze_store.setdefault(sid, {})['tracking_in_progress'] = True

    return redirect(url_for('eye_tracking', image=image))

@app.route('/eye_tracking_status')
def eye_tracking_status():
    sid = get_session_id()

    status = gaze_store.get(sid, {}).get('tracking_in_progress', False)

    return {'in_progress': status}

@app.route('/eye_tracking_results')
def eye_tracking_results():
    sid = get_session_id()

    gaze_store.setdefault(sid, {})['tracking_in_progress'] = False

    # Get all image gaze results for this session
    session_store = gaze_store.setdefault(sid, {})
    gaze_results_dict = session_store.get('gaze_results', {})  # {image_filename: gaze_data, ...}

    processed_images = list(gaze_results_dict.keys())

    selected_image = request.args.get('image')

    if not selected_image and processed_images:
        selected_image = processed_images[-1]
    elif not processed_images:
        selected_image = None

    # Generate and cache the heatmap only once per image
    heatmap_cache = session_store.setdefault('heatmap_cache', {})
    if selected_image not in heatmap_cache:
        gaze_data = gaze_results_dict.get(selected_image, [])

        bg_image_file = os.path.join(current_app.static_folder, selected_image)

        img = generate_heatmap(gaze_data, bg_image_file)

        img_io = io.BytesIO()
        img.save(img_io, 'PNG')

        heatmap_cache[selected_image] = img_io.getvalue()

    return render_template(
        'eye_tracking_results.html',
        processed_images=processed_images,
        selected_image=selected_image,
    )

@app.route('/gaze_heatmap_cached')
def gaze_heatmap_cached():
    sid = get_session_id()
    image = request.args.get('image')

    # Serve the already-generated PNG from cache
    cached = gaze_store.get(sid, {}).get('heatmap_cache', {}).get(image)
    if not cached:
        return ('Heatmap not generated', 404)

    return send_file(io.BytesIO(cached), mimetype='image/png')
