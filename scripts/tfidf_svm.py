"""
TF-IDF + SVM classifier for detecting misinformation-prone Reddit comments
via emotional-language signals.

Pipeline
--------
1.  Load labeled CSV (body_cleaned, label columns)
2.  TF-IDF vectorization  (unigrams + bigrams, sublinear TF, max 50 000 features)
3.  Optional emotional-language feature augmentation
4.  Linear SVM with hyperparameter search (GridSearchCV, 5-fold stratified)
5.  Evaluation: accuracy, precision, recall, F1, confusion matrix
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ── Emotional-language word banks ──────────────────────────────────────────────
# Derived from NRC / LIWC-style emotion categories commonly linked to
# misinformation spread (fear, anger, disgust, outrage).
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


def _emotion_features(texts: list[str]) -> np.ndarray:
    """Return a (n_samples, n_categories) matrix of per-category hit ratios."""
    categories = sorted(EMOTION_LEXICON)
    mat = np.zeros((len(texts), len(categories)), dtype=np.float32)
    for i, raw in enumerate(texts):
        tokens = set(re.findall(r"\b[a-z][a-z'-]+\b", raw.lower()))
        n_tok = max(len(tokens), 1)
        for j, cat in enumerate(categories):
            hits = tokens & EMOTION_LEXICON[cat]
            mat[i, j] = len(hits) / n_tok
    return mat


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_table(path: str) -> pd.DataFrame:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"No file at {p.resolve()}. Provide a valid path to a labeled CSV."
        )
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(p, lines=True)
    return pd.read_json(p)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train a TF-IDF + SVM model to classify Reddit comments "
        "as misinformation ('fake') or credible ('real')."
    )
    ap.add_argument("--input", required=True,
                    help="Path to labeled CSV / JSONL (needs text + label cols)")
    ap.add_argument("--text_col", default="body_cleaned",
                    help="Column with comment text (default: body_cleaned)")
    ap.add_argument("--label_col", default="label",
                    help="Column with ground-truth label (default: label)")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction held out for testing (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=50_000,
                    help="Max TF-IDF vocabulary size (default: 50000)")
    ap.add_argument("--ngram_max", type=int, default=2,
                    help="Max n-gram order (default: 2 = unigrams+bigrams)")
    ap.add_argument("--no_emotion_features", action="store_true",
                    help="Disable handcrafted emotion-lexicon features")
    ap.add_argument("--cv_folds", type=int, default=5,
                    help="Number of CV folds for hyperparameter search (default: 5)")
    ap.add_argument("--save_model", default=None,
                    help="If set, pickle the trained model to this path")
    ap.add_argument("--output_dir", default="results",
                    help="Directory to save plots and reports")
    args = ap.parse_args()

    # ── 1. Load & prepare data ─────────────────────────────────────────────
    print("Loading data …")
    df = load_table(args.input)[[args.text_col, args.label_col]].dropna()
    texts = df[args.text_col].astype(str).tolist()

    le = LabelEncoder()
    y = le.fit_transform(df[args.label_col])
    label_names = [str(c) for c in le.classes_]
    print(f"  {len(texts):,} samples  •  classes: {label_names}")
    for cls_name, cls_id in zip(label_names, range(len(label_names))):
        print(f"    {cls_name}: {(y == cls_id).sum():,}")

    strat = y if len(np.unique(y)) > 1 else None
    t_train, t_test, y_train, y_test = train_test_split(
        texts, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=strat,
    )
    print(f"  Train: {len(t_train):,}  •  Test: {len(t_test):,}\n")

    # ── 2. TF-IDF vectorization ────────────────────────────────────────────
    print("Fitting TF-IDF vectorizer …")
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=True,          # log(1 + tf) — dampens high-freq terms
        min_df=3,                   # ignore very rare tokens
        max_df=0.95,                # ignore near-universal tokens
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    X_train_tfidf = tfidf.fit_transform(t_train)
    X_test_tfidf = tfidf.transform(t_test)
    print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")

    # ── 3. Emotion features (optional) ─────────────────────────────────────
    if not args.no_emotion_features:
        print("Computing emotion-lexicon features …")
        E_train = _emotion_features(t_train)
        E_test = _emotion_features(t_test)
        X_train = hstack([X_train_tfidf, E_train])
        X_test = hstack([X_test_tfidf, E_test])
        print(f"  Emotion categories: {sorted(EMOTION_LEXICON.keys())}")
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf

    print(f"  Final feature matrix: {X_train.shape}\n")

    # ── 4. SVM with GridSearchCV ───────────────────────────────────────────
    print("Running GridSearchCV (LinearSVC) …")
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "loss": ["hinge", "squared_hinge"],
    }
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                         random_state=args.seed)
    grid = GridSearchCV(
        LinearSVC(max_iter=5000, random_state=args.seed, class_weight="balanced"),
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    print(f"  Search time : {elapsed:.1f}s\n")

    # ── 5. Evaluation on held-out test set ─────────────────────────────────
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred,
                                target_names=label_names, digits=4))

    # ── 6. Top features per class ──────────────────────────────────────────
    feature_names = tfidf.get_feature_names_out().tolist()
    if not args.no_emotion_features:
        feature_names += [f"EMO_{cat}" for cat in sorted(EMOTION_LEXICON)]

    # For binary LinearSVC: positive coefs → class 1 ("real"), negative → class 0 ("fake")
    coefs = best_model.coef_[0]
    top_k = 20
    # Most-negative coefficients → strongest predictors of "fake" (class 0)
    top_fake_idx = np.argsort(coefs)[:top_k]
    # Most-positive coefficients → strongest predictors of "real" (class 1)
    top_real_idx = np.argsort(coefs)[-top_k:][::-1]

    print(f"\nTop {top_k} features → '{label_names[0]}' (misinformation / fake):")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}")

    print(f"\nTop {top_k} features → '{label_names[1]}' (credible / real):")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}")

    # ── 7. Save results to file ───────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = out_dir / "tfidf_svm_results.txt"
    with open(report_path, "w") as f:
        f.write("TF-IDF + SVM RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=label_names, digits=4))
        f.write("\n\nTop Features (Fake):\n")
        for idx in top_fake_idx:
            f.write(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}\n")
        f.write("\nTop Features (Real):\n")
        for idx in top_real_idx:
            f.write(f"  {feature_names[idx]:30s}  {coefs[idx]:+.4f}\n")

    print(f"\nReport saved → {report_path.resolve()}")

    # ── 8. Plot Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: TF-IDF + SVM")
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = out_dir / "tfidf_svm_confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"Plot saved → {plot_path.resolve()}")

    # ── 9. Optionally save model ───────────────────────────────────────────
    if args.save_model:
        import joblib
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"tfidf": tfidf, "svm": best_model, "label_encoder": le}, model_path)
        print(f"Model saved → {model_path.resolve()}")


if __name__ == "__main__":
    main()
