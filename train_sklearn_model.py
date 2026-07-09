import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model_utils import (
    discover_dataset,
    load_audio,
    log_mel_features,
    save_manifest,
    sklearn_feature_vector_from_logmel,
)


def build_sklearn_features(rows, cache_path, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["x"].astype(np.float32), data["paths"].tolist()

    vectors = []
    paths = []
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        logmel = log_mel_features(load_audio(row["path"]))
        vectors.append(sklearn_feature_vector_from_logmel(logmel))
        paths.append(row["path"])
        if idx % 250 == 0 or idx == total:
            print(f"sklearn features {idx}/{total}", flush=True)

    x = np.stack(vectors).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, paths=np.asarray(paths, dtype=object))
    return x, paths


def decision_confidence(model, x):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.max(axis=1)
    return np.ones((x.shape[0],), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train sklearn closed-word pronunciation corrector.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="models")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, references, skipped = discover_dataset(args.dataset)
    words = sorted({row["word"] for row in rows}, key=lambda w: min(r["index"] for r in rows if r["word"] == w))
    word_to_id = {word: idx for idx, word in enumerate(words)}

    print(f"usable wavs: {len(rows)}", flush=True)
    print(f"skipped wavs/folders: {len(skipped)}", flush=True)
    print(f"words: {len(words)}", flush=True)
    save_manifest(out_dir / "manifest.json", rows, references, skipped, words)

    x, cached_paths = build_sklearn_features(rows, out_dir / "sklearn_features_cache.npz", args.force_cache)
    if cached_paths != [row["path"] for row in rows]:
        raise RuntimeError("Feature cache does not match current dataset. Rerun with --force-cache.")

    y_word = np.asarray([word_to_id[row["word"]] for row in rows], dtype=np.int64)
    y_quality = np.asarray([0 if row["quality"] == "D" else 1 for row in rows], dtype=np.int64)
    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.18,
        random_state=7,
        stratify=y_word,
    )

    word_model = make_pipeline(
        StandardScaler(),
        PCA(n_components=256, random_state=7),
        SVC(C=8.0, gamma="scale", class_weight="balanced"),
    )
    quality_model = make_pipeline(
        StandardScaler(),
        PCA(n_components=64, random_state=7),
        SVC(C=4.0, gamma="scale", class_weight="balanced"),
    )

    print("training validation word model...", flush=True)
    word_model.fit(x[train_idx], y_word[train_idx])
    pred_word = word_model.predict(x[val_idx])
    conf = decision_confidence(word_model, x[val_idx])

    print("training validation quality model...", flush=True)
    quality_model.fit(x[train_idx], y_quality[train_idx])
    pred_quality = quality_model.predict(x[val_idx])

    word_acc = float(accuracy_score(y_word[val_idx], pred_word))
    quality_acc = float(accuracy_score(y_quality[val_idx], pred_quality))
    print(f"validation word accuracy: {word_acc:.4f}", flush=True)
    print(f"validation D/N quality accuracy: {quality_acc:.4f}", flush=True)

    print("training final models on all usable data...", flush=True)
    word_model.fit(x, y_word)
    quality_model.fit(x, y_quality)

    artifact = {
        "model_type": "sklearn_svm_pca",
        "word_model": word_model,
        "quality_model": quality_model,
        "words": words,
        "word_to_id": word_to_id,
        "references": references,
    }
    model_path = out_dir / "sklearn_word_model.joblib"
    joblib.dump(artifact, model_path)

    summary = {
        "usable_wavs": len(rows),
        "skipped": skipped,
        "words": words,
        "validation_word_accuracy": word_acc,
        "validation_quality_accuracy": quality_acc,
        "validation_mean_confidence": float(conf.mean()),
        "classification_report": classification_report(
            y_word[val_idx],
            pred_word,
            target_names=words,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_word[val_idx], pred_word).tolist(),
    }
    with (out_dir / "sklearn_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"saved model: {model_path}", flush=True)


if __name__ == "__main__":
    main()
