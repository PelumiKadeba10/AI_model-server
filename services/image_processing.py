import cv2


def crop_face(
    image_path,
    face_location,
    output_path,
):
    image = cv2.imread(image_path)

    top, right, bottom, left = face_location

    face = image[top:bottom, left:right]

    resized = cv2.resize(face, (224, 224))

    cv2.imwrite(output_path, resized)

    return output_path