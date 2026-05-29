import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


_face_app: Optional[FaceAnalysis] = None

MODEL_NAME = "buffalo_l"
MODEL_ROOT = "/data/.insightface"

def _ensure_model_exists():
    """
    Triggers download ONLY if missing from persistent volume.
    """
    model_path = os.path.join(MODEL_ROOT, "models", MODEL_NAME)

    if not os.path.exists(model_path):
        os.makedirs(MODEL_ROOT, exist_ok=True)
        # This triggers download into /data
        FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT).prepare(ctx_id=-1)


def _get_face_app() -> FaceAnalysis:
    global _face_app

    if _face_app is None:
        os.makedirs(MODEL_ROOT, exist_ok=True)

        # ONLY downloads if volume is empty
        _ensure_model_exists()

        _face_app = FaceAnalysis(
            name=MODEL_NAME,
            root=MODEL_ROOT,
        )

        _face_app.prepare(ctx_id=-1, det_size=(640, 640))

    return _face_app


def get_faces(image: np.ndarray):
    app = _get_face_app()
    return app.get(image)


def get_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    faces = get_faces(image)
    if not faces:
        return None
    best_face = max(faces, key=lambda f: f.det_score)
    return np.asarray(best_face.normed_embedding, dtype=np.float32)


def compare_embeddings(emb1, emb2) -> float:
    a = np.asarray(emb1, dtype=np.float32)
    b = np.asarray(emb2, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def match_face(
    known_embeddings: Dict[str, np.ndarray],
    query_embedding,
    threshold: float = 0.6,
) -> Tuple[Optional[str], float]:
    if query_embedding is None or not known_embeddings:
        return None, 0.0

    best_name = None
    best_score = -1.0

    for name, embedding in known_embeddings.items():
        score = compare_embeddings(embedding, query_embedding)
        if score > best_score:
            best_name = name
            best_score = score

    if best_score < threshold:
        return None, best_score

    return best_name, best_score

