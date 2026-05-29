import os
from flask import Flask, request, jsonify
import uuid

from services.enrollment_pipeline import process_student_image
from services.face_engine import _get_face_app

app = Flask(__name__)


def init_model():
    os.makedirs("/data/.insightface", exist_ok=True)

    print("🚀 Loading Face Model...")
    _get_face_app()
    print("✅ Model ready!")


@app.route("/enroll", methods=["POST"])
def enroll():
    path = None

    try:
        image = request.files.get("image")
        matric_no = request.form.get("matric_no")
        index = request.form.get("index")

        if not image:
            return jsonify({"success": False, "message": "No image uploaded"}), 400

        os.makedirs("temp", exist_ok=True)

        path = f"temp/{uuid.uuid4()}.jpg"
        image.save(path)

        result = process_student_image(path, matric_no, index)

        if not result["success"]:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.route("/health", methods=["GET"])
def health():
    model_path = "/data/.insightface/models/buffalo_l"

    return jsonify({
        "success": True,
        "status": "AI model server is running",
        "checks": {
            "persistent_volume_mounted": os.path.exists("/data"),
            "buffalo_l_cached": os.path.exists(model_path)
        }
    }), 200


if __name__ == "__main__":
    init_model()   # 🔥 important for local runs
    app.run(debug=True)
    
init_model()