import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model_utils import SAMPLE_RATE, load_audio, log_mel_features, sklearn_feature_vector_from_logmel


class MuharaniMLP(nn.Module):
    def __init__(self, input_dim, num_words):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_words),
        )

    def forward(self, x):
        return self.net(x)


def load_features(path):
    return np.load(path, allow_pickle=True)["x"].astype(np.float32)


def evaluate(model, loader):
    model.eval()
    true = []
    pred = []
    conf = []
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.softmax(model(xb), dim=1)
            true.extend(yb.numpy().tolist())
            pred.extend(torch.argmax(probs, dim=1).numpy().tolist())
            conf.extend(torch.max(probs, dim=1).values.numpy().tolist())
    return np.asarray(true), np.asarray(pred), np.asarray(conf)


def quality_split_accuracy(true, pred, quality):
    quality = np.asarray(quality)
    metrics = {}
    for label in ("D", "N"):
        mask = quality == label
        metrics[label] = float(accuracy_score(true[mask], pred[mask])) if np.any(mask) else 0.0
    return metrics


def export_numpy_model(model, checkpoint, out_dir):
    state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    np.savez_compressed(
        out_dir / "mlp_word_model_np.npz",
        input_dim=np.asarray([checkpoint["input_dim"]], dtype=np.int64),
        words=np.asarray(checkpoint["words"], dtype=object),
        mean=checkpoint["mean"].astype(np.float32),
        std=checkpoint["std"].astype(np.float32),
        l1_w=state["net.0.weight"],
        l1_b=state["net.0.bias"],
        b1_w=state["net.1.weight"],
        b1_b=state["net.1.bias"],
        b1_mean=state["net.1.running_mean"],
        b1_var=state["net.1.running_var"],
        l2_w=state["net.4.weight"],
        l2_b=state["net.4.bias"],
        b2_w=state["net.5.weight"],
        b2_b=state["net.5.bias"],
        b2_mean=state["net.5.running_mean"],
        b2_var=state["net.5.running_var"],
        l3_w=state["net.8.weight"],
        l3_b=state["net.8.bias"],
        b3_w=state["net.9.weight"],
        b3_b=state["net.9.bias"],
        b3_mean=state["net.9.running_mean"],
        b3_var=state["net.9.running_var"],
        l4_w=state["net.12.weight"],
        l4_b=state["net.12.bias"],
    )


def repeat_d_indices(indices, q_base, rows_per_part, d_repeat):
    if d_repeat <= 1:
        return indices
    d_indices = indices[q_base[indices % rows_per_part] == "D"]
    return np.concatenate([indices] + [d_indices for _ in range(d_repeat - 1)])


def extract_reference_audio(audio, max_seconds=1.35):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size <= int((max_seconds + 0.25) * SAMPLE_RATE):
        return audio

    frame = int(0.025 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    energies = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        chunk = audio[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    if not energies:
        return audio[: int(max_seconds * SAMPLE_RATE)]

    energies = np.asarray(energies, dtype=np.float32)
    if energies.size >= 7:
        energies = np.convolve(energies, np.ones(7, dtype=np.float32) / 7.0, mode="same")
    floor = float(np.percentile(energies, 30))
    speech = np.maximum(energies - floor, 0.0)
    peak_idx = int(np.argmax(speech))
    peak_center = int(peak_idx * hop + frame // 2)
    window = int(max_seconds * SAMPLE_RATE)
    start = max(0, min(audio.size - window, peak_center - window // 2))
    return audio[start : start + window]


def export_reference_vectors(words, mean, std, out_dir):
    vectors = []
    word_ids = []
    references_dir = Path("static") / "references"
    for word_id, _ in enumerate(words):
        path = references_dir / f"{word_id}.wav"
        if not path.exists():
            continue
        audio = extract_reference_audio(load_audio(path))
        vector = sklearn_feature_vector_from_logmel(log_mel_features(audio)).reshape(1, -1)
        embedding = ((vector.astype(np.float32) - mean) / std).reshape(-1)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        vectors.append(embedding.astype(np.float32))
        word_ids.append(word_id)
    if not vectors:
        return None
    path = out_dir / "reference_vectors_mlp.npz"
    np.savez_compressed(
        path,
        vectors=np.stack(vectors).astype(np.float32),
        word_ids=np.asarray(word_ids, dtype=np.int64),
        words=np.asarray(words, dtype=object),
        space="mlp_standardized_cosine_app_window",
    )
    return path


def fit_production_model(x, y, q_base, words, references, mean, std, out_dir, epochs, batch_size, d_repeat):
    rows_per_part = len(q_base)
    train_idx = repeat_d_indices(np.arange(len(y)), q_base, rows_per_part, d_repeat)
    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MuharaniMLP(input_dim=x.shape[1], num_words=len(words))
    counts = np.bincount(y[train_idx], minlength=len(words)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            seen += xb.size(0)
        scheduler.step()
        epoch_loss = loss_sum / max(1, seen)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"production epoch {epoch:02d}/{epochs} loss={epoch_loss:.4f}", flush=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": x.shape[1],
        "words": words,
        "references": references,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "model_type": "muharani_mlp",
        "trained_on_all_data": True,
        "production_epochs": epochs,
        "d_repeat": d_repeat,
        "history": history,
    }
    model_path = out_dir / "mlp_word_model.pt"
    torch.save(checkpoint, model_path)
    export_numpy_model(model, checkpoint, out_dir)
    export_reference_vectors(words, mean, std, out_dir)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-repeat", type=int, default=2, help="Repeat incorrect/D training samples to improve correction recall")
    parser.add_argument("--production-only-epochs", type=int, default=0, help="Skip validation and train/export the final all-data model for N epochs")
    parser.add_argument("--skip-production-final", action="store_true", help="Only export the best validation checkpoint")
    args = parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)

    out_dir = Path(args.out)
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = manifest["rows"]
    words = manifest["words"]
    word_to_id = {word: idx for idx, word in enumerate(words)}
    y_base = np.asarray([word_to_id[row["word"]] for row in rows], dtype=np.int64)
    q_base = np.asarray([row["quality"] for row in rows], dtype=object)

    clean = load_features(out_dir / "sklearn_features_cache.npz")
    noise_paths = sorted(out_dir.glob("sklearn_noise_features_cache_*.npz"))
    noise_sets = [load_features(path) for path in noise_paths]
    x_parts = [clean] + noise_sets
    y_parts = [y_base for _ in x_parts]
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = ((x - mean) / std).astype(np.float32)

    if args.production_only_epochs > 0:
        history = fit_production_model(
            x,
            y,
            q_base,
            words,
            manifest.get("references", {}),
            mean,
            std,
            out_dir,
            args.production_only_epochs,
            args.batch_size,
            args.d_repeat,
        )
        summary_path = out_dir / "mlp_training_summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
        else:
            summary = {"words": words, "noise_copies": len(noise_sets), "d_repeat": args.d_repeat}
        summary.update(
            {
                "production_trained_on_all_data": True,
                "production_epochs": args.production_only_epochs,
                "production_history": history,
            }
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"saved production model: {out_dir / 'mlp_word_model.pt'}", flush=True)
        print(f"saved production numpy model: {out_dir / 'mlp_word_model_np.npz'}", flush=True)
        return

    base_indices = np.arange(len(rows))
    train_base, val_base = train_test_split(base_indices, test_size=0.18, random_state=7, stratify=y_base)
    train_idx = np.concatenate([train_base + part_idx * len(rows) for part_idx in range(len(x_parts))])
    train_idx = repeat_d_indices(train_idx, q_base, len(rows), args.d_repeat)
    val_clean_idx = val_base
    val_noise_idx = val_base + len(rows) if noise_sets else val_base

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]).long())
    val_clean_ds = TensorDataset(torch.from_numpy(x[val_clean_idx]), torch.from_numpy(y[val_clean_idx]).long())
    val_noise_ds = TensorDataset(torch.from_numpy(x[val_noise_idx]), torch.from_numpy(y[val_noise_idx]).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=args.batch_size)
    val_noise_loader = DataLoader(val_noise_ds, batch_size=args.batch_size)

    model = MuharaniMLP(input_dim=x.shape[1], num_words=len(words))
    counts = np.bincount(y[train_idx], minlength=len(words)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1.0
    history = []
    model_path = out_dir / "mlp_word_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            seen += xb.size(0)
        scheduler.step()

        clean_true, clean_pred, _ = evaluate(model, val_clean_loader)
        noise_true, noise_pred, _ = evaluate(model, val_noise_loader)
        clean_acc = float(accuracy_score(clean_true, clean_pred))
        noise_acc = float(accuracy_score(noise_true, noise_pred))
        clean_split = quality_split_accuracy(clean_true, clean_pred, q_base[val_base])
        noise_split = quality_split_accuracy(noise_true, noise_pred, q_base[val_base])
        score = (clean_acc + noise_acc) / 2.0
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / seen,
                "clean_acc": clean_acc,
                "clean_d_acc": clean_split["D"],
                "clean_n_acc": clean_split["N"],
                "noise_acc": noise_acc,
                "noise_d_acc": noise_split["D"],
                "noise_n_acc": noise_split["N"],
            }
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} loss={loss_sum/seen:.4f} "
            f"clean={clean_acc:.4f} clean_D={clean_split['D']:.4f} clean_N={clean_split['N']:.4f} "
            f"noise={noise_acc:.4f} noise_D={noise_split['D']:.4f} noise_N={noise_split['N']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": x.shape[1],
                    "words": words,
                    "references": manifest.get("references", {}),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                    "model_type": "muharani_mlp",
                    "best_epoch": epoch,
                    "history": history,
                },
                model_path,
            )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    clean_true, clean_pred, clean_conf = evaluate(model, val_clean_loader)
    noise_true, noise_pred, noise_conf = evaluate(model, val_noise_loader)
    clean_split = quality_split_accuracy(clean_true, clean_pred, q_base[val_base])
    noise_split = quality_split_accuracy(noise_true, noise_pred, q_base[val_base])
    summary = {
        "words": words,
        "epochs": args.epochs,
        "best_epoch": int(checkpoint.get("best_epoch", args.epochs)),
        "noise_copies": len(noise_sets),
        "d_repeat": args.d_repeat,
        "clean_accuracy": float(accuracy_score(clean_true, clean_pred)),
        "clean_d_accuracy": clean_split["D"],
        "clean_n_accuracy": clean_split["N"],
        "noise_accuracy": float(accuracy_score(noise_true, noise_pred)),
        "noise_d_accuracy": noise_split["D"],
        "noise_n_accuracy": noise_split["N"],
        "clean_mean_confidence": float(clean_conf.mean()),
        "noise_mean_confidence": float(noise_conf.mean()),
        "history": history,
        "classification_report_clean": classification_report(clean_true, clean_pred, target_names=words, zero_division=0, output_dict=True),
        "classification_report_noise": classification_report(noise_true, noise_pred, target_names=words, zero_division=0, output_dict=True),
    }
    if args.skip_production_final:
        export_numpy_model(model, checkpoint, out_dir)
        summary["production_trained_on_all_data"] = False
    else:
        production_epochs = int(checkpoint.get("best_epoch", args.epochs))
        summary["production_history"] = fit_production_model(
            x,
            y,
            q_base,
            words,
            manifest.get("references", {}),
            mean,
            std,
            out_dir,
            production_epochs,
            args.batch_size,
            args.d_repeat,
        )
        summary["production_trained_on_all_data"] = True
        summary["production_epochs"] = production_epochs
    with (out_dir / "mlp_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"saved model: {model_path}", flush=True)
    print(f"saved numpy model: {out_dir / 'mlp_word_model_np.npz'}", flush=True)
    print(f"best clean accuracy: {summary['clean_accuracy']:.4f}", flush=True)
    print(f"best clean D accuracy: {summary['clean_d_accuracy']:.4f}", flush=True)
    print(f"best clean N accuracy: {summary['clean_n_accuracy']:.4f}", flush=True)
    print(f"best noise accuracy: {summary['noise_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
