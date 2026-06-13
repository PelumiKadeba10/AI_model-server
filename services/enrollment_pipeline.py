import os
import cv2

from services.face_validation import (
    validate_face_exists,
    detect_multiple_faces,
    reject_blurry_image,
)
from services.image_processing import crop_face
from services.cloudflare_r2 import upload_file_to_r2
from services.embeddings import generate_embedding_from_faces

def process_student_image(image_path, matric_no, index, face_app):
    # 🎯 FIX 1: Read image from storage exactly ONCE into system memory
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": "Invalid image file format structure"}

    # STEP 1: Pass the image data matrix and the global app mapping method
    valid_face, face_locations, raw_faces = validate_face_exists(image, face_app.get)
    if not valid_face:
        return {"success": False, "error": face_locations}

    # STEP 2: Avoid reading file from disk again
    if not reject_blurry_image(image, threshold=100):
        return {"success": False, "error": "Image is blurry"}

    # STEP 3
    if not detect_multiple_faces(face_locations):
        return {"success": False, "error": "Multiple faces detected"}

    # Crop target region using extracted locations
    face_location = face_locations[0]
    os.makedirs("uploads", exist_ok=True)
    processed_path = f"uploads/processed_{matric_no}_{index}.jpg"
    
    crop_face(image_path, face_location, processed_path)

    # STEP 4: Store assets to Cloudflare R2
    remote_path = f"students/{matric_no}/{index}.jpg"
    cloudflare_url = upload_file_to_r2(processed_path, remote_path)

    # STEP 5: Zero inference penalty. Extract vectors directly from cached tokens
    embedding = generate_embedding_from_faces(raw_faces)
    if embedding is None:
        return {"success": False, "error": "Embedding extraction failed"}

    return {
        "success": True,
        "image_url": cloudflare_url,
        "embedding": embedding,
    }