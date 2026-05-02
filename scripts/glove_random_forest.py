import argparse
from operator import le
from pathlib import Path
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load data
def load_table(path):
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"No file at {p.resolve()}")
    return pd.read_csv(p)


# load glove
def load_glove(glove_path):
    print("Loading GloVe embeddings...")
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} words")
    return embeddings


# sentence to vector by averaging word vectors
def text_to_vector(text, embeddings, dim=100):
    words = text.split()
    vectors = [embeddings[w] for w in words if w in embeddings]

    if len(vectors) == 0:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to labeled CSV")
    parser.add_argument("--glove_path", required=True, help="Path to GloVe file")
    parser.add_argument("--text_col", default="body_cleaned")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # dataset
    df = load_table(args.input)[[args.text_col, args.label_col]].dropna()
    texts = df[args.text_col].astype(str).tolist()

    # encode
    le = LabelEncoder()
    y = le.fit_transform(df[args.label_col])

    #  load
    glove = load_glove(args.glove_path)

    # text to vector
    print("Converting text to vectors...")
    X = np.array([text_to_vector(t, glove) for t in texts])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # train random forest
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # eval
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print("\nResults:")
    print("Accuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    
    # save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "glove_rf_results.txt", "w") as f:
        f.write("GLOVE + RANDOM FOREST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Saved results to {output_dir / 'glove_rf_results.txt'}")
    
    # plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: GloVe + Random Forest")
    plt.colorbar()

    labels = le.classes_
    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # Add numbers on squares
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                 ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig(output_dir / "glove_rf_confusion_matrix.png")
    plt.close()
    


if __name__ == "__main__":
    main()