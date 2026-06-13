import cv2

def validate_face_exists(image, get_faces_fn):
    """
    Validates face existence using a pre-loaded image array and global engine function.
    """
    faces = get_faces_fn(image)

    if len(faces) == 0:
        return False, "No face detected", []

    # Store standard bounding box mapping format for cropping
    face_locations = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_locations.append((y1, x2, y2, x1))

    # Pass 'faces' object along so subsequent steps can use its pre-computed data
    return True, face_locations, faces


def detect_multiple_faces(face_locations):
    return len(face_locations) <= 1


def reject_blurry_image(image, threshold=100):
    """
    Uses the Laplacian variance method on a pre-loaded image matrix.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= threshold