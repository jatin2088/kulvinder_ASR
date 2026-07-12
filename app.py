import json
import csv
import os
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
REFERENCE_VECTOR_PATH = BASE_DIR / "models" / "reference_vectors_mlp.npz"
MANIFEST_PATH = BASE_DIR / "models" / "manifest.json"
DATA_DIR = BASE_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
FEEDBACK_VECTOR_DIR = DATA_DIR / "feedback_vectors"
RESULTS_JSONL = DATA_DIR / "results.jsonl"
RESULTS_CSV = DATA_DIR / "results.csv"
FEEDBACK_CSV = DATA_DIR / "feedback.csv"

app = Flask(__name__)

artifact = joblib.load(MODEL_PATH)
quality_model = artifact.get("quality_model")
references = artifact.get("references", {})
PREFERRED_WORD_MODEL = os.getenv("WORD_MODEL_KIND", "mlp").strip().lower()
MIN_ACCEPT_CONFIDENCE = float(os.getenv("MIN_ACCEPT_CONFIDENCE", "0.60"))
MLP_FALLBACK_BELOW = float(os.getenv("MLP_FALLBACK_BELOW", "0.60"))
REQUIRED_SAMPLES_PER_WORD = int(os.getenv("REQUIRED_SAMPLES_PER_WORD", "3"))
VOICE_MAP_ONLY = os.getenv("VOICE_MAP_ONLY", "0").strip().lower() not in {"0", "false", "no"}
VOICE_MAP_MIN_CONFIDENCE = float(os.getenv("VOICE_MAP_MIN_CONFIDENCE", "0.42"))
VOICE_MAP_MIN_MARGIN = float(os.getenv("VOICE_MAP_MIN_MARGIN", "0.08"))
MAX_WORD_SECONDS = float(os.getenv("MAX_WORD_SECONDS", "1.35"))
REFERENCE_BOOST_SIMILARITY = float(os.getenv("REFERENCE_BOOST_SIMILARITY", "0.55"))
REFERENCE_BOOST_MARGIN = float(os.getenv("REFERENCE_BOOST_MARGIN", "0.08"))
REFERENCE_BOOST_BLEND = float(os.getenv("REFERENCE_BOOST_BLEND", "0.55"))
REFERENCE_BOOST_TEMPERATURE = float(os.getenv("REFERENCE_BOOST_TEMPERATURE", "18.0"))
mlp_model = np.load(MLP_MODEL_PATH, allow_pickle=True) if MLP_MODEL_PATH.exists() else None
reference_vector_bank = np.load(REFERENCE_VECTOR_PATH, allow_pickle=True) if REFERENCE_VECTOR_PATH.exists() else None

if PREFERRED_WORD_MODEL == "mlp" and mlp_model is not None:
    model_kind = "mlp_numpy"
    word_model = None
    words = mlp_model["words"].tolist()
else:
    model_kind = "sklearn"
    word_model = artifact["word_model"]
    words = artifact["words"]

if MANIFEST_PATH.exists():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
        references.update(manifest.get("references", {}))

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_VECTOR_DIR.mkdir(parents=True, exist_ok=True)


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
        probs = mlp_probabilities(vector)
    else:
        scores = np.asarray(word_model.decision_function(vector), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        if mlp_model is not None and MLP_FALLBACK_BELOW > 0:
            low_confidence = probs.max(axis=1) < MLP_FALLBACK_BELOW
            if np.any(low_confidence):
                mlp_probs = mlp_probabilities(vector)
                probs[low_confidence] = mlp_probs[low_confidence]
    return apply_reference_boost(probs, vector)


def reference_query_embedding(vector):
    query = np.asarray(vector, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if mlp_model is not None:
        query = ((query - mlp_model["mean"]) / mlp_model["std"]).astype(np.float32)
    query = query[0]
    return query / (np.linalg.norm(query) + 1e-6)


def apply_reference_boost(probs, vector):
    if reference_vector_bank is None or REFERENCE_BOOST_BLEND <= 0:
        return probs
    vectors = reference_vector_bank["vectors"].astype(np.float32)
    word_ids = reference_vector_bank["word_ids"].astype(np.int64)
    if vectors.size == 0 or word_ids.size == 0:
        return probs

    query = reference_query_embedding(vector)
    sims = np.matmul(vectors, query)
    order = np.argsort(sims)[::-1]
    best = float(sims[order[0]])
    second = float(sims[order[1]]) if order.size > 1 else -1.0
    if best < REFERENCE_BOOST_SIMILARITY or best - second < REFERENCE_BOOST_MARGIN:
        return probs

    scores = np.exp((sims - sims.max()) * REFERENCE_BOOST_TEMPERATURE)
    ref_probs = np.zeros((len(words),), dtype=np.float64)
    ref_probs[word_ids] = scores / max(float(scores.sum()), 1e-9)
    boosted = (1.0 - REFERENCE_BOOST_BLEND) * probs[0] + REFERENCE_BOOST_BLEND * ref_probs
    return boosted.reshape(1, -1)


def feedback_samples():
    vectors = []
    labels = []
    for path in sorted(FEEDBACK_VECTOR_DIR.glob("*.npz")):
        try:
            data = np.load(path)
            vector = data["vector"].astype(np.float32).reshape(-1)
            word_id = int(np.asarray(data["word_id"]).reshape(-1)[0])
            vectors.append(vector)
            labels.append(word_id)
        except Exception:
            continue
    if not vectors or len(vectors) != len(labels):
        return None, None
    return np.stack(vectors), np.asarray(labels, dtype=np.int64)


def extract_word_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size <= int((MAX_WORD_SECONDS + 0.25) * SAMPLE_RATE):
        return audio

    frame = int(0.025 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    energies = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        chunk = audio[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    if not energies:
        return audio[: int(MAX_WORD_SECONDS * SAMPLE_RATE)]

    energies = np.asarray(energies, dtype=np.float32)
    if energies.size >= 7:
        energies = np.convolve(energies, np.ones(7, dtype=np.float32) / 7.0, mode="same")
    floor = float(np.percentile(energies, 30))
    speech = np.maximum(energies - floor, 0.0)
    peak_idx = int(np.argmax(speech))
    peak_center = int(peak_idx * hop + frame // 2)
    window = int(MAX_WORD_SECONDS * SAMPLE_RATE)
    start = max(0, min(audio.size - window, peak_center - window // 2))
    return audio[start : start + window]


def feature_vector_from_audio(audio):
    speech = extract_word_audio(audio)
    return sklearn_feature_vector_from_logmel(log_mel_features(speech)).reshape(1, -1), speech


def voice_map_status():
    _, labels = feedback_samples()
    counts = np.zeros((len(words),), dtype=np.int64)
    if labels is not None:
        counts = np.bincount(labels, minlength=len(words)).astype(np.int64)
    missing = [
        {"word_id": idx, "word": words[idx], "count": int(count)}
        for idx, count in enumerate(counts)
        if int(count) < REQUIRED_SAMPLES_PER_WORD
    ]
    complete = len(missing) == 0
    return {
        "complete": complete,
        "required_per_word": REQUIRED_SAMPLES_PER_WORD,
        "total_samples": int(counts.sum()),
        "required_total": int(REQUIRED_SAMPLES_PER_WORD * len(words)),
        "complete_words": int(np.sum(counts >= REQUIRED_SAMPLES_PER_WORD)),
        "words": [{"word_id": idx, "word": words[idx], "count": int(count)} for idx, count in enumerate(counts)],
        "missing": missing,
    }


def word_embedding(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if model_kind == "sklearn" and word_model is not None and hasattr(word_model, "steps"):
        return word_model[:-1].transform(vectors).astype(np.float32)
    return vectors.astype(np.float32)


def voice_map_probabilities(vector):
    feedback_vectors, labels = feedback_samples()
    if feedback_vectors is None:
        return None, 0, {"best_similarity": 0.0, "margin": 0.0}

    sample_count = len(labels)
    query = word_embedding(vector)
    bank = word_embedding(feedback_vectors)
    query_norm = query[0] / (np.linalg.norm(query[0]) + 1e-6)
    sample_norms = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-6)
    sims = np.matmul(sample_norms, query_norm)
    best = float(np.max(sims)) if sims.size else 0.0

    class_scores = np.zeros((len(words),), dtype=np.float64)
    top = np.argsort(sims)[-min(15, sample_count):]
    weights = np.exp((sims[top] - sims[top].max()) * 18.0)
    for idx, weight in zip(top, weights):
        class_scores[labels[idx]] += float(weight)

    for word_id in np.unique(labels):
        class_bank = sample_norms[labels == word_id]
        centroid = class_bank.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
        centroid_sim = float(np.dot(centroid, query_norm))
        class_scores[int(word_id)] += float(np.exp((centroid_sim - best) * 10.0)) * 0.35

    probs = class_scores / max(float(class_scores.sum()), 1e-9)
    order = np.argsort(probs)[::-1]
    top_conf = float(probs[order[0]]) if order.size else 0.0
    second_conf = float(probs[order[1]]) if order.size > 1 else 0.0
    meta = {"best_similarity": best, "margin": top_conf - second_conf}
    return probs.reshape(1, -1), sample_count, meta


def apply_feedback_probs(base_probs, vector):
    feedback_vectors, labels = feedback_samples()
    if feedback_vectors is None:
        return base_probs, 0, "generic", {"best_similarity": 0.0, "margin": 0.0}

    map_probs, sample_count, meta = voice_map_probabilities(vector)
    status = voice_map_status()
    if status["complete"]:
        return map_probs, sample_count, "voice_map", meta

    required_best = 0.88 if sample_count < 25 else 0.72 if sample_count < 100 else 0.62
    if meta["best_similarity"] < required_best:
        return base_probs, sample_count, "generic", meta

    blend = 0.78 if meta["best_similarity"] >= 0.88 else 0.55
    return (1.0 - blend) * base_probs + blend * map_probs, sample_count, "generic_feedback", meta


def top_predictions(vector, limit=3):
    probs, feedback_count, recognizer, meta = apply_feedback_probs(word_probabilities(vector), vector)
    order = np.argsort(probs[0])[::-1][:limit]
    return (
        [{"word_id": int(i), "word": words[int(i)], "confidence": float(probs[0, i])} for i in order],
        feedback_count,
        recognizer,
        meta,
    )


def audio_stats(audio):
    if audio.size == 0:
        return {"duration": 0.0, "rms": 0.0, "peak": 0.0}
    return {
        "duration": float(audio.size / SAMPLE_RATE),
        "rms": float(np.sqrt(np.mean(audio * audio) + 1e-12)),
        "peak": float(np.max(np.abs(audio))),
    }


def predict_wav(path):
    map_status = voice_map_status()
    if VOICE_MAP_ONLY and not map_status["complete"]:
        raise ValueError(
            f"Voice map incomplete: {map_status['total_samples']}/{map_status['required_total']} samples saved. "
            "Open Child Calibration and record the missing words first."
        )

    audio = load_audio(path)
    stats = audio_stats(audio)
    if stats["duration"] < 0.20 or stats["rms"] < 0.001:
        raise ValueError("Recording is too short or too quiet. Please speak closer to the phone.")

    vector, speech_audio = feature_vector_from_audio(audio)
    speech_stats = audio_stats(speech_audio)

    alternatives, feedback_count, recognizer, map_meta = top_predictions(vector, limit=3)
    word_id = alternatives[0]["word_id"]
    quality_id = int(quality_model.predict(vector)[0]) if quality_model is not None else 1

    confidence = alternatives[0]["confidence"]
    if recognizer == "voice_map":
        accepted = bool(confidence >= VOICE_MAP_MIN_CONFIDENCE and map_meta["margin"] >= VOICE_MAP_MIN_MARGIN)
    else:
        accepted = bool(confidence >= MIN_ACCEPT_CONFIDENCE)
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
        "recognizer": recognizer,
        "feedback_samples": feedback_count,
        "min_accept_confidence": MIN_ACCEPT_CONFIDENCE,
        "voice_map": map_status,
        "voice_map_margin": map_meta["margin"],
        "voice_map_similarity": map_meta["best_similarity"],
        "quality": "normal" if quality_id == 1 else "incorrect",
        "quality_label": "Normal / correct" if quality_id == 1 else "Incorrect / needs practice",
        "quality_confidence": estimate_confidence(quality_model, vector) if quality_model is not None else 0.0,
        "reference_url": f"/reference/{word_id}",
        "audio": {**stats, "speech_duration": speech_stats["duration"], "speech_rms": speech_stats["rms"]},
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
                "recognizer",
                "feedback_samples",
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


@app.get("/calibrate")
def calibrate():
    return render_template("calibrate.html", words=words)


@app.get("/health")
def health():
    status = voice_map_status()
    return jsonify(
        {
            "ok": True,
            "words": len(words),
            "model": model_kind,
            "preferred_model": PREFERRED_WORD_MODEL,
            "feedback_samples": status["total_samples"],
            "voice_map": status,
            "voice_map_only": VOICE_MAP_ONLY,
            "min_accept_confidence": MIN_ACCEPT_CONFIDENCE,
            "voice_map_min_confidence": VOICE_MAP_MIN_CONFIDENCE,
            "voice_map_min_margin": VOICE_MAP_MIN_MARGIN,
            "mlp_fallback_below": MLP_FALLBACK_BELOW if mlp_model is not None else 0.0,
        }
    )


@app.get("/map-status")
def map_status():
    return jsonify(voice_map_status())


@app.get("/words")
def word_list():
    return jsonify([{"word_id": idx, "word": word} for idx, word in enumerate(words)])


def save_feedback_vector(recording_id, word_id, recording_path):
    audio = load_audio(recording_path)
    vector, speech_audio = feature_vector_from_audio(audio)
    vector = vector.reshape(-1).astype(np.float32)
    vector_path = FEEDBACK_VECTOR_DIR / f"{recording_id}_{word_id}.npz"
    np.savez_compressed(vector_path, vector=vector, word_id=np.asarray([word_id], dtype=np.int64))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_CSV.exists()
    with FEEDBACK_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["created_at", "recording_id", "word_id", "word", "recording_path", "vector_path"])
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "recording_id": recording_id,
                "word_id": word_id,
                "word": words[word_id],
                "recording_path": str(recording_path),
                "vector_path": str(vector_path),
            }
        )
    return vector_path, audio_stats(speech_audio)


@app.post("/enroll")
def enroll():
    if "audio" not in request.files:
        return jsonify({"error": "missing audio file"}), 400
    word_id_raw = str(request.form.get("word_id", "")).strip()
    if not word_id_raw.isdigit():
        return jsonify({"error": "word_id is required"}), 400
    word_id = int(word_id_raw)
    if word_id < 0 or word_id >= len(words):
        return jsonify({"error": "bad word id"}), 400

    recording_id = uuid.uuid4().hex
    recording_path = RECORDINGS_DIR / f"{int(time.time())}_{recording_id}.wav"
    request.files["audio"].save(recording_path)

    try:
        audio = load_audio(recording_path)
        stats = audio_stats(audio)
        if stats["duration"] < 0.20 or stats["rms"] < 0.001:
            return jsonify({"error": "Recording is too short or too quiet."}), 400
        _, speech_stats = save_feedback_vector(recording_id, word_id, recording_path)
    except Exception as exc:
        return jsonify({"error": str(exc), "recording_id": recording_id}), 400

    status = voice_map_status()
    return jsonify(
        {
            "ok": True,
            "recording_id": recording_id,
            "word_id": word_id,
            "word": words[word_id],
            "feedback_samples": status["total_samples"],
            "voice_map": status,
            "speech_duration": speech_stats["duration"],
        }
    )


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
            "recognizer": "",
            "feedback_samples": "",
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
        "recognizer": result["recognizer"],
        "feedback_samples": result["feedback_samples"],
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


def find_recording(recording_id):
    if not recording_id.replace("-", "").isalnum():
        return None
    matches = sorted(RECORDINGS_DIR.glob(f"*_{recording_id}.wav"))
    return matches[-1] if matches else None


@app.post("/feedback")
def feedback():
    payload = request.get_json(silent=True) or request.form
    recording_id = str(payload.get("recording_id", "")).strip()
    word_id_raw = str(payload.get("word_id", "")).strip()
    if not recording_id or not word_id_raw.isdigit():
        return jsonify({"error": "recording_id and word_id are required"}), 400

    word_id = int(word_id_raw)
    if word_id < 0 or word_id >= len(words):
        return jsonify({"error": "bad word id"}), 400

    recording_path = find_recording(recording_id)
    if recording_path is None:
        return jsonify({"error": "recording not found"}), 404

    save_feedback_vector(recording_id, word_id, recording_path)
    status = voice_map_status()

    return jsonify(
        {
            "ok": True,
            "recording_id": recording_id,
            "word_id": word_id,
            "word": words[word_id],
            "feedback_samples": status["total_samples"],
            "voice_map": status,
        }
    )


@app.get("/recordings/<recording_id>")
def recording(recording_id):
    recording_path = find_recording(recording_id)
    if recording_path is None:
        return jsonify({"error": "recording not found"}), 404
    return send_file(recording_path, mimetype="audio/wav", as_attachment=True, download_name=f"{recording_id}.wav")


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
