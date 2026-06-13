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


def generate_embedding_from_faces(faces):
    """
    Extracts the pre-computed embedding from the already identified face object.
    Reduces engine inference counts to exactly one per transaction.
    """
    if not faces:
        return None
    
    # Grab the top face object's normalized math array
    first_face = faces[0]
    
    if hasattr(first_face, 'normed_embedding') and first_face.normed_embedding is not None:
        return first_face.normed_embedding.tolist()
        
    if hasattr(first_face, 'embedding') and first_face.embedding is not None:
        return first_face.embedding.tolist()
        
    return None