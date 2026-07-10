import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
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

    base_indices = np.arange(len(rows))
    train_base, val_base = train_test_split(base_indices, test_size=0.18, random_state=7, stratify=y_base)
    train_idx = np.concatenate([train_base + part_idx * len(rows) for part_idx in range(len(x_parts))])
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
        score = (clean_acc + noise_acc) / 2.0
        history.append({"epoch": epoch, "loss": loss_sum / seen, "clean_acc": clean_acc, "noise_acc": noise_acc})
        print(f"epoch {epoch:02d}/{args.epochs} loss={loss_sum/seen:.4f} clean={clean_acc:.4f} noise={noise_acc:.4f}", flush=True)
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
                    "history": history,
                },
                model_path,
            )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    clean_true, clean_pred, clean_conf = evaluate(model, val_clean_loader)
    noise_true, noise_pred, noise_conf = evaluate(model, val_noise_loader)
    summary = {
        "words": words,
        "epochs": args.epochs,
        "noise_copies": len(noise_sets),
        "clean_accuracy": float(accuracy_score(clean_true, clean_pred)),
        "noise_accuracy": float(accuracy_score(noise_true, noise_pred)),
        "clean_mean_confidence": float(clean_conf.mean()),
        "noise_mean_confidence": float(noise_conf.mean()),
        "history": history,
        "classification_report_clean": classification_report(clean_true, clean_pred, target_names=words, zero_division=0, output_dict=True),
        "classification_report_noise": classification_report(noise_true, noise_pred, target_names=words, zero_division=0, output_dict=True),
    }
    with (out_dir / "mlp_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"saved model: {model_path}", flush=True)
    print(f"best clean accuracy: {summary['clean_accuracy']:.4f}", flush=True)
    print(f"best noise accuracy: {summary['noise_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
