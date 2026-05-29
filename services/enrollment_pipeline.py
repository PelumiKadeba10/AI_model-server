import os

from services.face_validation import (
    validate_face_exists,
    detect_multiple_faces,
    reject_blurry_image,
)

from services.image_processing import (
    crop_face,
)

from services.cloudflare_r2 import (
    upload_file_to_r2,
)

from services.embeddings import (
    generate_embedding,
)


def process_student_image(
    image_path,
    matric_no,
    index,
):
    # STEP 1
    valid_face, result = validate_face_exists(
        image_path
    )

    if not valid_face:
        return {
            "success": False,
            "error": result,
        }

    # STEP 2
    if not reject_blurry_image(image_path):
        return {
            "success": False,
            "error": "Image is blurry",
        }

    # STEP 3
    if not detect_multiple_faces(result):
        return {
            "success": False,
            "error": "Multiple faces detected",
        }

    face_location = result[0]
    
    os.makedirs("uploads", exist_ok=True)

    # STEP 4
    processed_path = (
        f"uploads/processed_{matric_no}_{index}.jpg"
    )

    crop_face(
        image_path,
        face_location,
        processed_path,
    )

    # STEP 5
    remote_path = (
        f"students/{matric_no}/{index}.jpg"
    )

    cloudflare_url = upload_file_to_r2(
        processed_path,
        remote_path,
    )

    # STEP 6
    embedding = generate_embedding(
        image_path
    )

    if embedding is None:
        return {
            "success": False,
            "error": "Embedding generation failed",
        }

    return {
        "success": True,
        "image_url": cloudflare_url,
        "embedding": embedding,
    }