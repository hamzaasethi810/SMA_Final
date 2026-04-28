import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_table(path):
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"No file at {p.resolve()}. Use an actual path to your labeled CSV."
        )
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(p, lines=True)
    return pd.read_json(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to your labeled CSV or JSONL")
    ap.add_argument("--text_col", default="body_cleaned")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--sbert", default="all-MiniLM-L6-v2")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--output_dir", default="results", help="Directory to save plots and reports")
    args = ap.parse_args()

    df = load_table(args.input)[[args.text_col, args.label_col]].dropna()
    texts = df[args.text_col].astype(str).tolist()
    le = LabelEncoder()
    y = le.fit_transform(df[args.label_col]).astype(np.int64)

    strat = y if len(np.unique(y)) > 1 else None
    t_train, t_test, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=strat
    )

    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    enc = SentenceTransformer(args.sbert, device=device)
    X_train = enc.encode(t_train, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    X_test = enc.encode(t_test, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    n_classes = len(le.classes_)
    model = MLP(X_train.shape[1], n_classes, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

    names = [str(c) for c in le.classes_]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=names, digits=4)
    
    print("accuracy:", acc)
    print("confusion_matrix:\n", cm)
    print(rep)

    # ── Save Results ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = out_dir / "sbert_mlp_results.txt"
    with open(report_path, "w") as f:
        f.write("SBERT + MLP RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(rep)
    print(f"Report saved → {report_path.resolve()}")

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot Confusion Matrix
    plt.subplot(1, 2, 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: SBERT+MLP")
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = out_dir / "sbert_mlp_plots.png"
    plt.savefig(plot_path)
    print(f"Plots saved → {plot_path.resolve()}")


if __name__ == "__main__":
    main()
