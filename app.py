import os
import uuid
from flask import Flask, request, jsonify

from services.enrollment_pipeline import process_student_image
from services.face_engine import _get_face_app  # Assuming this manages your model

app = Flask(__name__)

# Unified configuration for your Fly.dev volume mount
MOUNT_DIR = "/data"
MODEL_DIR = os.path.join(MOUNT_DIR, ".insightface")

# Create the cache directory structure if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Bootstrapping global application components...")

# ✅ Initialize the model ONCE globally when the server boots.
# Ensure your local `_get_face_app()` internally routes to the target path:
# e.g., FaceAnalysis(name='buffalo_l', root='/data/.insightface')
print("🤖 Loading heavy Face Model 'buffalo_l' into system RAM...")
GLOBAL_FACE_APP = _get_face_app() 
print("✅ Model loaded and ready to serve requests!")


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

        # 🎯 CRITICAL CHANGE: Pass your globally warmed-up model instance into the pipeline
        # update your enrollment_pipeline function signature to accept this parameter!
        result = process_student_image(path, matric_no, index, face_app=GLOBAL_FACE_APP)

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.route("/health", methods=["GET"])
def health():
    target_cache_path = os.path.join(MODEL_DIR, "models", "buffalo_l")
    
    return jsonify({
        "success": True,
        "status": "AI model server is online",
        "checks": {
            "persistent_volume_mounted": os.path.exists(MOUNT_DIR),
            "buffalo_l_cached": os.path.exists(target_cache_path)
        }
    }), 200


if __name__ == "__main__":
    # In production (Gunicorn/Uvicorn), this block is skipped, but globals load at the top.
    # For local testing, this runs perfectly.
    app.run(host="0.0.0.0", port=5000, debug=True)