import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from model_utils import load_audio, log_mel_features, sklearn_feature_vector_from_logmel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sklearn_word_model.joblib"
MANIFEST_PATH = BASE_DIR / "models" / "manifest.json"

app = Flask(__name__)

artifact = joblib.load(MODEL_PATH)
word_model = artifact["word_model"]
quality_model = artifact.get("quality_model")
words = artifact["words"]
references = artifact.get("references", {})

if MANIFEST_PATH.exists():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
        references.update(manifest.get("references", {}))


def estimate_confidence(model, vector):
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(vector), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return float(probs[0].max())
    return 1.0


def predict_wav(path):
    audio = load_audio(path)
    logmel = log_mel_features(audio)
    vector = sklearn_feature_vector_from_logmel(logmel).reshape(1, -1)

    word_id = int(word_model.predict(vector)[0])
    quality_id = int(quality_model.predict(vector)[0]) if quality_model is not None else 1

    return {
        "word_id": word_id,
        "word": words[word_id],
        "confidence": estimate_confidence(word_model, vector),
        "quality": "normal" if quality_id == 1 else "incorrect",
        "quality_label": "Normal / correct" if quality_id == 1 else "Incorrect / needs practice",
        "quality_confidence": estimate_confidence(quality_model, vector) if quality_model is not None else 0.0,
        "reference_url": f"/reference/{word_id}",
    }


@app.get("/")
def index():
    return render_template("index.html", words=words)


@app.get("/health")
def health():
    return jsonify({"ok": True, "words": len(words)})


@app.post("/predict")
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "missing audio file"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename.lower().endswith(".wav"):
        return jsonify({"error": "audio must be WAV"}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        audio_file.save(tmp.name)
        result = predict_wav(tmp.name)
    return jsonify(result)


@app.get("/reference/<int:word_id>")
def reference(word_id):
    if word_id < 0 or word_id >= len(words):
        return jsonify({"error": "bad word id"}), 404
    static_ref = BASE_DIR / "static" / "references" / f"{word_id}.wav"
    if static_ref.exists():
        return send_file(static_ref, mimetype="audio/wav", as_attachment=False)
    path = references.get(words[word_id])
    if not path:
        return jsonify({"error": "reference audio not found"}), 404
    ref_path = Path(path)
    if not ref_path.is_absolute():
        ref_path = BASE_DIR / ref_path
    return send_file(ref_path, mimetype="audio/wav", as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
