import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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
    ap.add_argument("--output_dir", default="results", help="Directory to save reports")
    ap.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations for Logistic Regression")
    args = ap.parse_args()

    # Load and clean data
    df = load_table(args.input)[[args.text_col, args.label_col]].dropna()
    texts = df[args.text_col].astype(str).tolist()
    le = LabelEncoder()
    y = le.fit_transform(df[args.label_col])

    # Split data
    strat = y if len(np.unique(y)) > 1 else None
    t_train, t_test, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=strat
    )

    # Device selection
    import torch
    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    # Encode texts
    print(f"Encoding texts using {args.sbert} on {device}...")
    enc = SentenceTransformer(args.sbert, device=device)
    X_train = enc.encode(t_train, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    X_test = enc.encode(t_test, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)

    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=args.max_iter, random_state=args.seed)
    model.fit(X_train, y_train)

    # Predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Metrics
    names = [str(c) for c in le.classes_]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=names, digits=4)
    
    print("\n" + "=" * 30)
    print("accuracy:", acc)
    print("confusion_matrix:\n", cm)
    print(rep)

    # Save Results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = out_dir / "sbert_LR_results.txt"
    with open(report_path, "w") as f:
        f.write("SBERT + LR RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(rep)
    print(f"Report saved -> {report_path.resolve()}")

    # ── Save Plots ───────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: SBERT+LR")
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
    
    plot_path = out_dir / "sbert_LR_plots.png"
    plt.savefig(plot_path)
    print(f"Plots saved -> {plot_path.resolve()}")

if __name__ == "__main__":
    main()
