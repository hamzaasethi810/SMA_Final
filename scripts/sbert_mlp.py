import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    args = ap.parse_args()

    df = load_table(args.input)[[args.text_col, args.label_col]].dropna()
    texts = df[args.text_col].astype(str).tolist()
    le = LabelEncoder()
    y = le.fit_transform(df[args.label_col]).astype(np.int64)

    strat = y if len(np.unique(y)) > 1 else None
    t_train, t_test, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=strat
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

    names = [str(c) for c in le.classes_]
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=names, digits=4))


if __name__ == "__main__":
    main()
