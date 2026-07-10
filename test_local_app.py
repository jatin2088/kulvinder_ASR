import io
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import app as web
from model_utils import SAMPLE_RATE, load_audio


def wav_bytes_from_audio(audio, sample_rate=SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    data = (audio * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, data)
    buffer.seek(0)
    return buffer


def post_wav(client, audio, target_word_id=None):
    buffer = wav_bytes_from_audio(audio)
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    data = {
        "audio": (buffer, "speech.wav"),
        "duration": f"{len(audio) / SAMPLE_RATE:.4f}",
        "rms": f"{rms:.6f}",
        "noise_floor": "0.012000",
    }
    if target_word_id is not None:
        data["target_word_id"] = str(target_word_id)
    return client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data",
    )


def add_noise(audio, snr_db=12.0):
    rng = np.random.default_rng(7)
    signal_power = float(np.mean(audio * audio) + 1e-12)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def main():
    # Keep the smoke test deterministic even if local calibration files exist.
    web.feedback_samples = lambda: (None, None)
    client = web.app.test_client()
    health = client.get("/health")
    assert health.status_code == 200, health.data
    assert health.json["ok"] is True

    reference = client.get("/reference/0")
    assert reference.status_code == 200
    assert reference.content_type == "audio/wav"

    with open("models/manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)

    rows = manifest["rows"]
    random.seed(7)
    by_word = {}
    for row in rows:
        by_word.setdefault(row["word"], []).append(row)
    sample_rows = [random.choice(items) for _, items in sorted(by_word.items())]
    clean_free_ok = 0
    noisy_free_ok = 0
    noisy_accepted = 0
    noisy_accepted_correct = 0

    for row in sample_rows:
        audio = load_audio(row["path"])

        clean_response = post_wav(client, audio)
        assert clean_response.status_code == 200, clean_response.data
        clean = clean_response.json
        clean_free_ok += int(clean["word"] == row["word"])

        noisy_audio = add_noise(audio)
        noisy_response = post_wav(client, noisy_audio)
        assert noisy_response.status_code == 200, noisy_response.data
        noisy = noisy_response.json
        noisy_free_ok += int(noisy["word"] == row["word"])
        noisy_accepted += int(noisy.get("accepted") is True)
        noisy_accepted_correct += int(noisy.get("accepted") is True and noisy["word"] == row["word"])


    quiet_response = post_wav(client, np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32))
    assert quiet_response.status_code == 400

    assert Path("data/results.csv").exists()
    assert Path("data/results.jsonl").exists()

    print(f"free clean sample accuracy: {clean_free_ok}/{len(sample_rows)}")
    print(f"free noisy sample accuracy: {noisy_free_ok}/{len(sample_rows)}")
    print(f"noisy accepted correct: {noisy_accepted_correct}/{noisy_accepted}")
    print("health/reference/save/error checks passed")


if __name__ == "__main__":
    main()
