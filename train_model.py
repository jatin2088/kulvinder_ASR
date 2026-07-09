import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model_utils import (
    N_MELS,
    SAMPLE_RATE,
    TARGET_SECONDS,
    discover_dataset,
    load_audio,
    log_mel_features,
    save_manifest,
)


class WordCorrectionNet(nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.input_norm = nn.LayerNorm(N_MELS)
        self.gru = nn.GRU(
            input_size=N_MELS,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20,
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4, 192),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.word_head = nn.Linear(192, num_words)
        self.quality_head = nn.Linear(192, 2)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = self.input_norm(x)
        seq, _ = self.gru(x)
        mean_pool = seq.mean(dim=1)
        max_pool = seq.max(dim=1).values
        z = self.proj(torch.cat([mean_pool, max_pool], dim=1))
        return self.word_head(z), self.quality_head(z)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_feature_cache(rows, cache_path, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["x"].astype(np.float32), data["paths"].tolist()

    features = []
    paths = []
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        audio = load_audio(row["path"])
        features.append(log_mel_features(audio))
        paths.append(row["path"])
        if idx % 250 == 0 or idx == total:
            print(f"features {idx}/{total}", flush=True)

    x = np.stack(features).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, paths=np.asarray(paths, dtype=object))
    return x, paths


def make_loaders(x, y_word, y_quality, train_idx, val_idx, batch_size):
    x_train = torch.from_numpy(x[train_idx]).unsqueeze(1)
    x_val = torch.from_numpy(x[val_idx]).unsqueeze(1)
    yw_train = torch.from_numpy(y_word[train_idx]).long()
    yw_val = torch.from_numpy(y_word[val_idx]).long()
    yq_train = torch.from_numpy(y_quality[train_idx]).long()
    yq_val = torch.from_numpy(y_quality[val_idx]).long()

    train_ds = TensorDataset(x_train, yw_train, yq_train)
    val_ds = TensorDataset(x_val, yw_val, yq_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    word_true, word_pred, word_prob = [], [], []
    quality_true, quality_pred = [], []
    with torch.no_grad():
        for xb, yw, yq in loader:
            xb = xb.to(device)
            logits_word, logits_quality = model(xb)
            probs = torch.softmax(logits_word, dim=1)
            word_true.extend(yw.numpy().tolist())
            word_pred.extend(torch.argmax(logits_word, dim=1).cpu().numpy().tolist())
            word_prob.extend(torch.max(probs, dim=1).values.cpu().numpy().tolist())
            quality_true.extend(yq.numpy().tolist())
            quality_pred.extend(torch.argmax(logits_quality, dim=1).cpu().numpy().tolist())

    word_true = np.asarray(word_true)
    word_pred = np.asarray(word_pred)
    quality_true = np.asarray(quality_true)
    quality_pred = np.asarray(quality_pred)
    return {
        "word_accuracy": float((word_true == word_pred).mean()),
        "quality_accuracy": float((quality_true == quality_pred).mean()),
        "mean_confidence": float(np.mean(word_prob)),
        "word_true": word_true,
        "word_pred": word_pred,
        "quality_true": quality_true,
        "quality_pred": quality_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Train closed-vocabulary Punjabi word correction model.")
    parser.add_argument("--dataset", default="dataset", help="Dataset folder")
    parser.add_argument("--out", default="models", help="Output model folder")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, references, skipped = discover_dataset(args.dataset)
    words = sorted({row["word"] for row in rows}, key=lambda w: min(r["index"] for r in rows if r["word"] == w))
    word_to_id = {word: idx for idx, word in enumerate(words)}

    print(f"usable wavs: {len(rows)}")
    print(f"skipped wavs/folders: {len(skipped)}")
    print(f"words: {len(words)}")

    save_manifest(out_dir / "manifest.json", rows, references, skipped, words)

    x, cached_paths = build_feature_cache(rows, out_dir / "features_cache.npz", force=args.force_cache)
    if cached_paths != [row["path"] for row in rows]:
        raise RuntimeError("Feature cache does not match current dataset. Rerun with --force-cache.")

    y_word = np.asarray([word_to_id[row["word"]] for row in rows], dtype=np.int64)
    y_quality = np.asarray([0 if row["quality"] == "D" else 1 for row in rows], dtype=np.int64)

    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.18,
        random_state=args.seed,
        stratify=y_word,
    )

    train_loader, val_loader = make_loaders(x, y_word, y_quality, train_idx, val_idx, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordCorrectionNet(num_words=len(words)).to(device)

    class_counts = np.bincount(y_word[train_idx], minlength=len(words)).astype(np.float32)
    word_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    word_weights = word_weights / word_weights.mean()
    quality_counts = np.bincount(y_quality[train_idx], minlength=2).astype(np.float32)
    quality_weights = quality_counts.sum() / np.maximum(quality_counts, 1.0)
    quality_weights = quality_weights / quality_weights.mean()

    criterion_word = nn.CrossEntropyLoss(weight=torch.tensor(word_weights, dtype=torch.float32).to(device))
    criterion_quality = nn.CrossEntropyLoss(weight=torch.tensor(quality_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_acc = -1.0
    best_path = out_dir / "word_correction_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yw, yq in train_loader:
            xb = xb.to(device)
            yw = yw.to(device)
            yq = yq.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits_word, logits_quality = model(xb)
            loss = criterion_word(logits_word, yw) + 0.25 * criterion_quality(logits_quality, yq)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / len(train_idx)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_word_accuracy": metrics["word_accuracy"],
                "val_quality_accuracy": metrics["quality_accuracy"],
                "val_mean_confidence": metrics["mean_confidence"],
            }
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"loss={train_loss:.4f} "
            f"word_acc={metrics['word_accuracy']:.4f} "
            f"quality_acc={metrics['quality_accuracy']:.4f}",
            flush=True,
        )

        if metrics["word_accuracy"] > best_acc:
            best_acc = metrics["word_accuracy"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "words": words,
                    "word_to_id": word_to_id,
                    "sample_rate": SAMPLE_RATE,
                    "target_seconds": TARGET_SECONDS,
                    "n_mels": N_MELS,
                    "references": references,
                    "model_class": "WordCorrectionNet",
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    final_metrics = evaluate(model, val_loader, device)

    report = classification_report(
        final_metrics["word_true"],
        final_metrics["word_pred"],
        target_names=words,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(final_metrics["word_true"], final_metrics["word_pred"]).tolist()

    summary = {
        "usable_wavs": len(rows),
        "skipped": skipped,
        "words": words,
        "best_val_word_accuracy": best_acc,
        "final_val_word_accuracy": final_metrics["word_accuracy"],
        "final_val_quality_accuracy": final_metrics["quality_accuracy"],
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"saved model: {best_path}")
    print(f"best validation word accuracy: {best_acc:.4f}")
    print(f"final quality D/N accuracy: {final_metrics['quality_accuracy']:.4f}")


if __name__ == "__main__":
    main()
