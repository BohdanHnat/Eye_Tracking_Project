import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, # Track 1 face only
    refine_landmarks=True, # Enable iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Positions of calibration targets
TARGET_POSITIONS = [
    "TOP_LEFT", "TOP_CENTER", "TOP_RIGHT",
    "CENTER_LEFT", "CENTER", "CENTER_RIGHT",
    "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"
]

CALIB_FRAMES = 70  # Frames per calibration target

ET_RECORD_FRAMES = 700 # Frames per eye-tracking image

CAM_FPS = 70 # Camera frames-per-second

LEFT_IRIS = [474, 475, 476, 477] # Left iris landmarks
RIGHT_IRIS = [469, 470, 471, 472] # Right iris landmarks
