import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_DEFAULT_NRC = Path(__file__).parent / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# ensures the NRC lexicon is available
def _ensure_nrc(path: Path) -> Path:
    if path.is_file():
        return path

    raise FileNotFoundError(
        f"NRC lexicon not found at '{path}'.\n"
        "Download it from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm "
        f"and place the TSV file at '{path}', or pass --nrc_path <file>."
    )


def load_nrc(path: Path) -> dict[str, set]:
    target = {"anger", "fear", "joy", "sadness"}
    p = _ensure_nrc(path)

    with open(p, encoding="utf-8") as f:
        first = f.readline().strip()

    parts = first.split("\t")

    if len(parts) == 3:
        lexicon = {e: set() for e in target}
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) != 3:
                    continue
                word, emotion, assoc = cols
                if emotion in target and assoc == "1":
                    lexicon[emotion].add(word.lower())
    else:
        df = pd.read_csv(p, sep="\t", header=0)
        word_col = df.columns[0]
        lexicon = {e: set() for e in target}
        for emotion in target:
            if emotion not in df.columns:
                raise ValueError(f"Column '{emotion}' not found in NRC file. Columns: {list(df.columns)}")
            lexicon[emotion] = set(df.loc[df[emotion] == 1, word_col].str.lower())

    sizes = {e: len(v) for e, v in lexicon.items()}
    print(f"NRC lexicon loaded from {p.name}: {sizes}")
    return lexicon


def _load_spacy(model: str):
    try:
        import spacy
        return spacy.load(model, disable=["parser", "ner"])
    except OSError:
        raise OSError(
            f"spaCy model '{model}' not found. "
            f"Run: python -m spacy download {model}"
        )


def _emotion_features(texts: list[str], lexicon: dict, nlp) -> np.ndarray:
    categories = sorted(lexicon)
    mat = np.zeros((len(texts), len(categories)), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
        n_tok = max(len(lemmas), 1)
        for j, cat in enumerate(categories):
            mat[i, j] = sum(1 for lem in lemmas if lem in lexicon[cat]) / n_tok
    return mat

def load_table(path):
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"No file at {p.resolve()}. Use to labeled CSV."
        )
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(p, lines=True)
    return pd.read_json(p)

def main():
    parser = argparse.ArgumentParser(
        description="Emotion enhanced hybrid: SBERT embeddings + emotion scores -> LogisticRegression"
    )
    parser.add_argument("--input", required=True, help="Path to labeled CSV or JSONL")
    parser.add_argument("--text_col", default="body_cleaned")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--sbert", default="all-MiniLM-L6-v2")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--no_emotion_features", action="store_true",
                        help="Disable emotion features (ablation)")
    parser.add_argument("--nrc_path", default=str(_DEFAULT_NRC),
                        help="Path to NRC Emotion Lexicon TSV. "
                             "downloaded via nrclex if the file does not exist.")
    parser.add_argument("--spacy_model", default="en_core_web_sm",
                        help="spaCy model for lemmatization (default: en_core_web_sm)")
    args = parser.parse_args()

    lexicon = load_nrc(Path(args.nrc_path))
    nlp = _load_spacy(args.spacy_model)

    df = load_table(args.input)[[args.text_col, args.label_col]].dropna()
    texts = df[args.text_col].astype(str).tolist()
    le = LabelEncoder()
    y = le.fit_transform(df[args.label_col])
    label_names = [str(c) for c in le.classes_]
    print(f"{len(texts):,} samples  •  classes: {label_names}")

    strat = y if len(np.unique(y)) > 1 else None
    t_train, t_test, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=strat
    )

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Encoding with {args.sbert} on {device}...")
    enc = SentenceTransformer(args.sbert, device=device)
    X_train = enc.encode(t_train, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    X_test = enc.encode(t_test, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)

    if not args.no_emotion_features:
        categories = sorted(lexicon)
        print(f"Computing emotion features: {categories}")
        E_train = _emotion_features(t_train, lexicon, nlp)
        E_test = _emotion_features(t_test, lexicon, nlp)
        X_train = np.concatenate([X_train, E_train], axis=1)
        X_test = np.concatenate([X_test, E_test], axis=1)
        print(f"Feature dim: {X_train.shape[1]} (SBERT + {len(categories)} emotion scores)")
    else:
        print(f"Feature dim: {X_train.shape[1]} (SBERT only, ablation mode)")

    print("Training...")
    model = LogisticRegression(max_iter=args.max_iter, random_state=args.seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=label_names, digits=4)

    print(f"\nAccuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print(rep)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "eeh_results.txt"
    with open(report_path, "w") as f:
        f.write("EMOTION ENHANCED HYBRID RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(rep)
    print(f"Report saved -> {report_path.resolve()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix: SBERT + Emotion + LR")
    tick_marks = np.arange(len(label_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(label_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    ax2 = axes[1]
    if not args.no_emotion_features:
        categories = sorted(lexicon)
        E_all = _emotion_features(texts, lexicon, nlp)
        y_all = le.transform(df[args.label_col])
        x = np.arange(len(categories))
        width = 0.35
        for k, name in enumerate(label_names):
            mask = y_all == k
            means = E_all[mask].mean(axis=0)
            ax2.bar(x + k * width, means, width, label=name)
        ax2.set_title("Mean Emotion Score by Class")
        ax2.set_xticks(x + width / 2)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel("Hit Ratio")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Emotion features disabled\n(ablation)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Emotion Scores (ablation)")

    plt.tight_layout()
    plot_path = out_dir / "eeh_plots.png"
    plt.savefig(plot_path)
    print(f"Plots saved -> {plot_path.resolve()}")

if __name__ == "__main__":
    main()
