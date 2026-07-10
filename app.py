import json
import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from model_utils import SAMPLE_RATE, load_audio, log_mel_features, sklearn_feature_vector_from_logmel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sklearn_word_model.joblib"
MLP_MODEL_PATH = BASE_DIR / "models" / "mlp_word_model_np.npz"
MANIFEST_PATH = BASE_DIR / "models" / "manifest.json"
DATA_DIR = BASE_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
RESULTS_JSONL = DATA_DIR / "results.jsonl"
RESULTS_CSV = DATA_DIR / "results.csv"

app = Flask(__name__)

artifact = joblib.load(MODEL_PATH)
quality_model = artifact.get("quality_model")
references = artifact.get("references", {})
if MLP_MODEL_PATH.exists():
    model_kind = "mlp_numpy"
    mlp_model = np.load(MLP_MODEL_PATH, allow_pickle=True)
    word_model = None
    words = mlp_model["words"].tolist()
else:
    model_kind = "sklearn"
    mlp_model = None
    word_model = artifact["word_model"]
    words = artifact["words"]

if MANIFEST_PATH.exists():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
        references.update(manifest.get("references", {}))

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


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


def silu(x):
    return x / (1.0 + np.exp(-x))


def linear(x, weight, bias):
    return np.matmul(x, weight.T) + bias


def batch_norm(x, gamma, beta, mean, var):
    return (x - mean) / np.sqrt(var + 1e-5) * gamma + beta


def mlp_probabilities(vector):
    x = ((vector.astype(np.float32) - mlp_model["mean"]) / mlp_model["std"]).astype(np.float32)
    x = silu(batch_norm(linear(x, mlp_model["l1_w"], mlp_model["l1_b"]), mlp_model["b1_w"], mlp_model["b1_b"], mlp_model["b1_mean"], mlp_model["b1_var"]))
    x = silu(batch_norm(linear(x, mlp_model["l2_w"], mlp_model["l2_b"]), mlp_model["b2_w"], mlp_model["b2_b"], mlp_model["b2_mean"], mlp_model["b2_var"]))
    x = silu(batch_norm(linear(x, mlp_model["l3_w"], mlp_model["l3_b"]), mlp_model["b3_w"], mlp_model["b3_b"], mlp_model["b3_mean"], mlp_model["b3_var"]))
    scores = linear(x, mlp_model["l4_w"], mlp_model["l4_b"]).astype(np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    return probs / probs.sum(axis=1, keepdims=True)


def word_probabilities(vector):
    if model_kind == "mlp_numpy":
        return mlp_probabilities(vector)
    scores = np.asarray(word_model.decision_function(vector), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    return probs / probs.sum(axis=1, keepdims=True)


def top_predictions(vector, limit=3):
    probs = word_probabilities(vector)
    order = np.argsort(probs[0])[::-1][:limit]
    return [{"word_id": int(i), "word": words[int(i)], "confidence": float(probs[0, i])} for i in order]


def audio_stats(audio):
    if audio.size == 0:
        return {"duration": 0.0, "rms": 0.0, "peak": 0.0}
    return {
        "duration": float(audio.size / SAMPLE_RATE),
        "rms": float(np.sqrt(np.mean(audio * audio) + 1e-12)),
        "peak": float(np.max(np.abs(audio))),
    }


def predict_wav(path):
    audio = load_audio(path)
    stats = audio_stats(audio)
    if stats["duration"] < 0.20 or stats["rms"] < 0.001:
        raise ValueError("Recording is too short or too quiet. Please speak closer to the phone.")

    logmel = log_mel_features(audio)
    vector = sklearn_feature_vector_from_logmel(logmel).reshape(1, -1)

    alternatives = top_predictions(vector, limit=3)
    word_id = alternatives[0]["word_id"]
    quality_id = int(quality_model.predict(vector)[0]) if quality_model is not None else 1

    confidence = alternatives[0]["confidence"]
    accepted = bool(confidence >= 0.20)
    return {
        "word_id": word_id,
        "word": words[word_id],
        "confidence": confidence,
        "accepted": accepted,
        "alternatives": alternatives,
        "model_guess_word_id": word_id,
        "model_guess_word": words[word_id],
        "model_guess_confidence": confidence,
        "model_kind": model_kind,
        "quality": "normal" if quality_id == 1 else "incorrect",
        "quality_label": "Normal / correct" if quality_id == 1 else "Incorrect / needs practice",
        "quality_confidence": estimate_confidence(quality_model, vector) if quality_model is not None else 0.0,
        "reference_url": f"/reference/{word_id}",
        "audio": stats,
    }


def append_result(row):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "created_at",
                "word",
                "confidence",
                "quality",
                "quality_confidence",
                "duration",
                "rms",
                "peak",
                "client_duration",
                "client_rms",
                "client_noise_floor",
                "model_guess_word",
                "model_guess_confidence",
                "alternatives",
                "accepted",
                "model_kind",
                "recording_path",
                "user_agent",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


@app.get("/")
def index():
    return render_template("index.html", words=words)


@app.get("/health")
def health():
    return jsonify({"ok": True, "words": len(words), "model": model_kind})


@app.post("/predict")
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "missing audio file"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename.lower().endswith(".wav"):
        return jsonify({"error": "audio must be WAV"}), 400

    recording_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    recording_path = RECORDINGS_DIR / f"{int(time.time())}_{recording_id}.wav"

    try:
        audio_file.save(recording_path)
        result = predict_wav(recording_path)
    except Exception as exc:
        error_row = {
            "id": recording_id,
            "created_at": created_at,
            "word": "",
            "confidence": "",
            "quality": "error",
            "quality_confidence": "",
            "duration": "",
            "rms": "",
            "peak": "",
            "client_duration": request.form.get("duration", ""),
            "client_rms": request.form.get("rms", ""),
            "client_noise_floor": request.form.get("noise_floor", ""),
            "model_guess_word": "",
            "model_guess_confidence": "",
            "alternatives": "",
            "accepted": "",
            "model_kind": model_kind,
            "recording_path": str(recording_path),
            "user_agent": request.headers.get("User-Agent", ""),
            "error": str(exc),
        }
        append_result(error_row)
        return jsonify({"error": str(exc), "recording_id": recording_id}), 400

    row = {
        "id": recording_id,
        "created_at": created_at,
        "word": result["word"],
        "confidence": round(result["confidence"], 6),
        "quality": result["quality"],
        "quality_confidence": round(result["quality_confidence"], 6),
        "duration": round(result["audio"]["duration"], 4),
        "rms": round(result["audio"]["rms"], 6),
        "peak": round(result["audio"]["peak"], 6),
        "client_duration": request.form.get("duration", ""),
        "client_rms": request.form.get("rms", ""),
        "client_noise_floor": request.form.get("noise_floor", ""),
        "model_guess_word": result["model_guess_word"],
        "model_guess_confidence": round(result["model_guess_confidence"], 6),
        "alternatives": json.dumps(result["alternatives"], ensure_ascii=False),
        "accepted": result["accepted"],
        "model_kind": result["model_kind"],
        "recording_path": str(recording_path),
        "user_agent": request.headers.get("User-Agent", ""),
    }
    append_result(row)
    result["recording_id"] = recording_id
    result["recording_url"] = f"/recordings/{recording_id}"
    return jsonify(result)


@app.get("/results.csv")
def results_csv():
    if not RESULTS_CSV.exists():
        return jsonify({"error": "no results saved yet"}), 404
    return send_file(RESULTS_CSV, mimetype="text/csv", as_attachment=True, download_name="results.csv")


@app.get("/recordings/<recording_id>")
def recording(recording_id):
    if not recording_id.replace("-", "").isalnum():
        return jsonify({"error": "bad recording id"}), 404
    matches = sorted(RECORDINGS_DIR.glob(f"*_{recording_id}.wav"))
    if not matches:
        return jsonify({"error": "recording not found"}), 404
    return send_file(matches[-1], mimetype="audio/wav", as_attachment=True, download_name=f"{recording_id}.wav")


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
