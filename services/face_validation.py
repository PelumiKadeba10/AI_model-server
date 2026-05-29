import cv2
from services.face_engine import get_faces


def validate_face_exists(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False, "Invalid image"

    faces = get_faces(image)

    if len(faces) == 0:
        return False, "No face detected"

    face_locations = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_locations.append((y1, x2, y2, x1))

    return True, face_locations



def detect_multiple_faces(face_locations):
    if len(face_locations) > 1:
        return False

    return True



def reject_blurry_image(image_path, threshold=100):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return variance >= threshold
