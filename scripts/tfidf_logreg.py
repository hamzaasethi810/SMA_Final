"""
TF-IDF + Logistic Regression classifier for detecting misinformation-prone
Reddit comments via emotional-language signals.

Pipeline
--------
1. Load labeled CSV with body_cleaned and label columns
2. TF-IDF vectorization using unigrams + bigrams
3. Add emotional-language features
4. Train Logistic Regression classifier
5. Evaluate with accuracy, confusion matrix, classification report
6. Print top predictive features
"""

import matplotlib.pyplot as plt
import argparse
import re
import time
from pathlib import Path


import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# ── Emotional-language word banks ──────────────────────────────────────────────
EMOTION_LEXICON = {
    "fear": {
        "afraid", "alarm", "anxiety", "apprehensive", "dread", "fear",
        "fright", "horror", "panic", "scared", "terrified", "terror",
        "threat", "worried", "nightmare", "phobia",
    },
    "anger": {
        "angry", "bitter", "enraged", "furious", "hate", "hatred",
        "hostile", "irate", "livid", "mad", "outrage", "outraged",
        "rage", "resent", "resentment", "wrath", "infuriating",
    },
    "disgust": {
        "abhor", "appalling", "detest", "disgust", "disgusted",
        "disgusting", "gross", "loathe", "nauseating", "repulsive",
        "revolting", "sick", "sickening", "vile", "repugnant",
    },
    "sensationalism": {
        "amazing", "bombshell", "breaking", "devastating", "epic",
        "exclusive", "explosive", "incredible", "insane", "jaw-dropping",
        "massive", "mindblowing", "outrageous", "shocking", "stunning",
        "unbelievable", "unprecedented", "urgent", "you won't believe",
    },
    "absolutism": {
        "absolutely", "always", "certainly", "completely", "definitely",
        "entirely", "every", "everything", "never", "nobody", "nothing",
        "obviously", "perfectly", "totally", "undeniably", "without a doubt",
    },
}


def emotion_features(texts: list[str]) -> np.ndarray:
    """
    Creates emotion-based features for each text.

    Each row becomes a small set of numbers showing how much fear, anger,
    disgust, sensationalism, and absolutism language appears in that comment.
    """
    categories = sorted(EMOTION_LEXICON)
    mat = np.zeros((len(texts), len(categories)), dtype=np.float32)

    for i, raw in enumerate(texts):
        tokens = set(re.findall(r"\b[a-z][a-z'-]+\b", raw.lower()))
        n_tokens = max(len(tokens), 1)

        for j, category in enumerate(categories):
            hits = tokens & EMOTION_LEXICON[category]
            mat[i, j] = len(hits) / n_tokens

    return mat


def load_table(path: str) -> pd.DataFrame:
    """
    Loads the input file.
    The team mainly uses CSV, but this also supports JSON/JSONL.
    """
    p = Path(path).expanduser()

    if not p.is_file():
        raise FileNotFoundError(
            f"No file found at {p.resolve()}. Please provide a valid path."
        )

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)

    if p.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(p, lines=True)

    return pd.read_json(p)


def main():
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + Logistic Regression model to classify "
                    "Reddit comments as fake or real."
    )

    parser.add_argument("--input", required=True,
                        help="Path to labeled CSV / JSONL")
    parser.add_argument("--text_col", default="body_cleaned",
                        help="Column with comment text")
    parser.add_argument("--label_col", default="label",
                        help="Column with fake/real label")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data used for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50000,
                        help="Max TF-IDF vocabulary size")
    parser.add_argument("--ngram_max", type=int, default=2,
                        help="Max n-gram order")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save results")

    args = parser.parse_args()

    # ── 1. Load and prepare data ───────────────────────────────────────────
    print("Loading data ...")
    df = load_table(args.input)[[args.text_col, args.label_col]].dropna()

    texts = df[args.text_col].astype(str).tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[args.label_col])
    label_names = [str(c) for c in label_encoder.classes_]

    print(f"  {len(texts):,} samples")
    print(f"  Classes: {label_names}")

    for class_name, class_id in zip(label_names, range(len(label_names))):
        print(f"    {class_name}: {(y == class_id).sum():,}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    print(f"  Train: {len(X_train_text):,}")
    print(f"  Test : {len(X_test_text):,}\n")

    # ── 2. TF-IDF vectorization ────────────────────────────────────────────
    print("Fitting TF-IDF vectorizer ...")

    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
    )

    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")

    # ── 3. Emotion features ────────────────────────────────────────────────
    print("Computing emotion-lexicon features ...")

    E_train = emotion_features(X_train_text)
    E_test = emotion_features(X_test_text)

    X_train = hstack([X_train_tfidf, E_train])
    X_test = hstack([X_test_tfidf, E_test])

    emotion_categories = sorted(EMOTION_LEXICON.keys())

    print(f"  Emotion categories: {emotion_categories}")
    print(f"  Final feature matrix: {X_train.shape}\n")

    # ── 4. Logistic Regression model ───────────────────────────────────────
    print("Training Logistic Regression ...")

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=args.seed
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"  Training time: {elapsed:.1f}s\n")

    # ── 5. Evaluation ──────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    print("=" * 60)
    print("TEST SET RESULTS: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print()

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    print(classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        digits=4
    ))

        # ── 6. Plot Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: TF-IDF + Logistic Regression")
    plt.colorbar()
    plt.grid(False)

    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    # Add numbers inside boxes
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save image
    plot_path = Path(args.output_dir) / "tfidf_logreg_confusion_matrix.png"
    plt.savefig(plot_path)

    print(f"Confusion matrix saved → {plot_path.resolve()}")

    # ── 7. Top features per class ──────────────────────────────────────────
    feature_names = tfidf.get_feature_names_out().tolist()
    feature_names += [f"EMO_{cat}" for cat in emotion_categories]

    coefs = model.coef_[0]
    top_k = 20

    top_fake_idx = np.argsort(coefs)[:top_k]
    top_real_idx = np.argsort(coefs)[-top_k:][::-1]

    print(f"\nTop {top_k} features → '{label_names[0]}' (fake):")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}")

    print(f"\nTop {top_k} features → '{label_names[1]}' (real):")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}")

    # ── 8. Save results ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "tfidf_logreg_results.txt"

    with open(report_path, "w") as f:
        f.write("TF-IDF + LOGISTIC REGRESSION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            digits=4
        ))

        f.write(f"\nTop {top_k} Features: {label_names[0]} / fake\n")
        for idx in top_fake_idx:
            f.write(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}\n")

        f.write(f"\nTop {top_k} Features: {label_names[1]} / real\n")
        for idx in top_real_idx:
            f.write(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}\n")

    print(f"\nReport saved → {report_path.resolve()}")


if __name__ == "__main__":
    main()