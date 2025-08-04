import cv2
import mediapipe as mp
from screeninfo import get_monitors

# Get the resolution of the primary monitor
monitor = get_monitors()[0]
w, h = monitor.width, monitor.height

CAM_FPS = 60

# Camera and face_mesh setup
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, CAM_FPS) # FPS
camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) # Width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h) # Height
camera.set(cv2.CAP_PROP_CONTRAST, 50) # Contrast
camera.set(cv2.CAP_PROP_BRIGHTNESS, 100) # Brightness

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,             # Track 1 face only
    refine_landmarks=True,       # Enable iris landmarks
    min_detection_confidence=0.5, # Face detection threshold
    min_tracking_confidence=0.5   # Tracking confidence threshold
)

# Use your existing positions list
TARGET_POSITIONS = [
    "TOP_LEFT", "TOP_CENTER", "TOP_RIGHT",
    "CENTER_LEFT", "CENTER", "CENTER_RIGHT",
    "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"
]

CALIB_FRAMES = 70  # frames per calibration target

ET_RECORD_FRAMES = 200 # frames per eye tracking image

RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]