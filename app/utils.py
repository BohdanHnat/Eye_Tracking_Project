from flask import session, current_app
import uuid
import os

gaze_store = {}

def get_session_id():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return session['sid']

def get_image_files():
    images_folder = os.path.join(current_app.static_folder, 'images')
    try:
        files = [
            f for f in os.listdir(images_folder)
            if f.startswith('img_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'))
        ]
        return sorted(files)
    except FileNotFoundError:
        return []