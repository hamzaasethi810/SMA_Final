import argparse
from pathlib import Path

import pandas as pd


def read_subreddit_set(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments_bz2", required=True, type=str)
    parser.add_argument("--out_csv", required=True, type=str)
    parser.add_argument("--text_col", default="body_cleaned", type=str)
    parser.add_argument("--label_mode", default="subreddit_lists", choices=["subreddit_lists", "banned_vs_not"], type=str)
    parser.add_argument("--fake_subreddits", default=None, type=str)
    parser.add_argument("--real_subreddits", default=None, type=str)
    parser.add_argument("--subreddits_metadata_jsonl", default=None, type=str)
    parser.add_argument("--max_rows", default=200000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    usecols = [args.text_col, "subreddit", "score"]
    chunks = pd.read_json(
        args.comments_bz2,
        compression="bz2",
        lines=True,
        dtype=False,
        chunksize=100000,
    )

    fake_set = read_subreddit_set(args.fake_subreddits)
    real_set = read_subreddit_set(args.real_subreddits)

    banned = set()
    if args.label_mode == "banned_vs_not":
        if not args.subreddits_metadata_jsonl:
            raise ValueError("--subreddits_metadata_jsonl is required for label_mode=banned_vs_not")
        meta = pd.read_json(args.subreddits_metadata_jsonl, lines=True)
        banned = set(meta.loc[meta["banned"] == 1, "subreddit"].astype(str))

    out_rows = []
    seen = 0
    for c in chunks:
        c = c[[col for col in usecols if col in c.columns]].dropna(subset=[args.text_col, "subreddit"])
        if args.label_mode == "subreddit_lists":
            c = c[c["subreddit"].isin(fake_set | real_set)]
            if len(c) == 0:
                continue
            c["label"] = c["subreddit"].map(lambda s: "fake" if s in fake_set else "real")
        else:
            c["label"] = c["subreddit"].map(lambda s: "fake" if s in banned else "real")

        remaining = args.max_rows - seen
        if remaining <= 0:
            break
        if len(c) > remaining:
            c = c.iloc[:remaining]
        out_rows.append(c[[args.text_col, "label", "score", "subreddit"]])
        seen += len(c)
        if seen >= args.max_rows:
            break

    if not out_rows:
        raise ValueError("No rows matched your labeling rule. Check your subreddit lists / metadata path.")

    out = pd.concat(out_rows, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

