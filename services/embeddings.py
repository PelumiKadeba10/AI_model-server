import cv2

from services.face_engine import get_embedding as _engine_get_embedding
from services.face_engine import get_faces


def detect_faces(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return []
    return get_faces(image)


def detect_multiple_faces(faces) -> bool:
    return len(faces) > 1


def generate_embedding(image_path: str):
    """
    Returns:
        List[float] | None
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    embedding = _engine_get_embedding(image)
    if embedding is None:
        return None
    return embedding.tolist()


def validate_face_exists(image_path: str):
    faces = detect_faces(image_path)
    if not faces:
        return False, "No face detected"
    return True, faces


def is_blurry(image_path: str, threshold: float = 80.0) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        return True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

