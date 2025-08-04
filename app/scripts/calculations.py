import numpy as np

# Get the average position of iris landmarks (centroid = pupil center)
def get_iris_center(landmarks, indexes, img_w, img_h):
    points = [landmarks[i] for i in indexes]
    x = int(sum(p.x for p in points) / len(points) * img_w)
    y = int(sum(p.y for p in points) / len(points) * img_h)
    return (x, y)

def normalize_iris(iris_point, eye_inner, eye_outer):
    eye_width = np.linalg.norm(np.array(eye_outer) - np.array(eye_inner))
    eye_center = np.mean([eye_inner, eye_outer], axis=0)
    rel_x = (iris_point[0] - eye_center[0]) / eye_width
    rel_y = (iris_point[1] - eye_center[1]) / eye_width
    # Flip X so that X increases left to right
    rel_x = -rel_x
    # Y already increases from top to bottom in OpenCV
    return [rel_x, rel_y]

