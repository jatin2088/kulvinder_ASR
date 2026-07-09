import json
import math
import re
import warnings
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from scipy.signal import resample_poly, stft


SAMPLE_RATE = 16000
TARGET_SECONDS = 2.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SECONDS)
N_MELS = 48
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 320


def parse_folder_name(folder_name):
    match = re.match(r"^\s*(\d+)\s+(.+)\s+([DN])\s*$", folder_name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected dataset folder name: {folder_name}")
    index = int(match.group(1))
    word = match.group(2).strip()
    quality = match.group(3).upper()
    return index, word, quality


def discover_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    rows = []
    references = {}
    skipped = []

    for folder in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        try:
            index, word, quality = parse_folder_name(folder.name)
        except ValueError as exc:
            skipped.append({"path": str(folder), "reason": str(exc)})
            continue

        for wav_path in sorted(folder.glob("*.wav")):
            ok, reason = is_valid_wav(wav_path)
            if not ok:
                skipped.append({"path": str(wav_path), "reason": reason})
                continue
            rows.append(
                {
                    "path": str(wav_path),
                    "index": index,
                    "word": word,
                    "quality": quality,
                    "folder": folder.name,
                }
            )
            if quality == "N" and word not in references:
                references[word] = str(wav_path)

    rows.sort(key=lambda r: (r["index"], r["word"], r["quality"], r["path"]))
    return rows, references, skipped


def is_valid_wav(path):
    path = Path(path)
    if path.stat().st_size < 1000:
        return False, "too small / broken wav"
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getframerate() <= 0 or handle.getnframes() <= 0:
                return False, "empty wav"
            duration = handle.getnframes() / handle.getframerate()
            if duration <= 0 or duration > 10:
                return False, f"bad duration {duration:.2f}s"
            if handle.getnchannels() != 1:
                return False, "not mono"
            if handle.getsampwidth() != 2:
                return False, "not 16-bit pcm"
    except Exception as exc:
        return False, f"cannot read wav: {exc}"
    return True, ""


def load_audio(path, sample_rate=SAMPLE_RATE):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, audio = wavfile.read(str(path))
    audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    if sr != sample_rate:
        gcd = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, sr // gcd).astype(np.float32)

    return audio.astype(np.float32)


def normalize_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(TARGET_SAMPLES, dtype=np.float32)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-5:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def trim_silence(audio, min_keep=0.25):
    audio = normalize_audio(audio)
    if audio.size == 0:
        return audio
    frame = 320
    hop = 160
    energies = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        chunk = audio[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    if not energies:
        return audio
    energies = np.asarray(energies)
    threshold = max(0.015, float(np.percentile(energies, 75)) * 0.2)
    active = np.where(energies > threshold)[0]
    if active.size == 0:
        return audio
    start = max(0, int(active[0] * hop - 0.08 * SAMPLE_RATE))
    end = min(audio.size, int(active[-1] * hop + frame + 0.18 * SAMPLE_RATE))
    trimmed = audio[start:end]
    if trimmed.size < int(min_keep * SAMPLE_RATE):
        return audio
    return trimmed


def fit_audio_length(audio, target_samples=TARGET_SAMPLES):
    audio = trim_silence(audio)
    if audio.size > target_samples:
        start = max(0, (audio.size - target_samples) // 2)
        audio = audio[start : start + target_samples]
    elif audio.size < target_samples:
        pad_left = (target_samples - audio.size) // 2
        pad_right = target_samples - audio.size - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode="constant")
    return normalize_audio(audio)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=50.0,
    fmax=7600.0,
):
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, filters.shape[1])):
            filters[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, min(right, filters.shape[1])):
            filters[m - 1, k] = (right - k) / float(right - center)
    return filters


_MEL_FILTERS = mel_filterbank()


def log_mel_features(audio):
    audio = fit_audio_length(audio)
    _, _, spec = stft(
        audio,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    power = np.abs(spec).astype(np.float32) ** 2
    mel = np.matmul(_MEL_FILTERS, power)
    logmel = np.log10(np.maximum(mel, 1e-8))
    logmel = (logmel - float(logmel.mean())) / (float(logmel.std()) + 1e-6)
    return logmel.astype(np.float32)


def sklearn_feature_vector_from_logmel(logmel):
    mfcc = dct(logmel, type=2, axis=0, norm="ortho")[:24, :]
    delta = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])
    delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])

    def summarize(arr):
        return np.concatenate(
            [
                arr.mean(axis=1),
                arr.std(axis=1),
                arr.max(axis=1),
                arr.min(axis=1),
                np.percentile(arr, 25, axis=1),
                np.percentile(arr, 75, axis=1),
            ]
        )

    downsampled = mfcc[:, ::3].reshape(-1)
    vector = np.concatenate([summarize(mfcc), summarize(delta), summarize(delta2), downsampled])
    return vector.astype(np.float32)


def sklearn_feature_vector_from_audio(audio):
    return sklearn_feature_vector_from_logmel(log_mel_features(audio))


def save_manifest(path, rows, references, skipped, words):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_rate": SAMPLE_RATE,
                "target_seconds": TARGET_SECONDS,
                "n_mels": N_MELS,
                "words": words,
                "references": references,
                "rows": rows,
                "skipped": skipped,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def dataset_summary(rows):
    by_word = defaultdict(lambda: {"D": 0, "N": 0})
    for row in rows:
        by_word[row["word"]][row["quality"]] += 1
    return dict(sorted(by_word.items(), key=lambda item: item[0]))
