"""
Prepare HC3 (English, QA part) for the Chapter 6 classification fine-tuning
in "Build a Large Language Model (From Scratch)".

Output files have exactly the same format as the book's SMS spam CSVs:
    columns: Label, Text      (Label: 0 = human, 1 = AI/ChatGPT)
so they can be fed directly into the book's SpamDataset class.

Data source
-----------
The original HC3 lives on Hugging Face (Hello-SimpleAI/HC3). The HC3-Plus repo
on GitHub (suu990901/chatgpt-comparison-detection-HC3-Plus) re-hosts the
English HC3 QA answers as flat {"text", "label"} jsonl files:
    data/en/val_hc3_QA.jsonl   (5,851 rows)
    data/en/test_hc3_QA.jsonl  (25,049 rows)
These two files are pure HC3 QA (reddit_eli5, open_qa, wiki_csai, finance,
medicine), so we pool them and re-split ourselves. (HC3-Plus's train.jsonl
mixes HC3 with summarization / translation / paraphrase data, so we skip it.)

Usage
-----
    pip install pandas nltk scikit-learn tiktoken
    python prepare_hc3.py --out_dir .            # downloads from GitHub
    python prepare_hc3.py --raw_dir path/to/data/en --out_dir .   # local copy
"""

import argparse
import json
import os
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

RAW_BASE = ("https://raw.githubusercontent.com/suu990901/"
            "chatgpt-comparison-detection-HC3-Plus/main/data/en/")
RAW_FILES = ["val_hc3_QA.jsonl", "test_hc3_QA.jsonl"]

_detok = TreebankWordDetokenizer()


# --------------------------------------------------------------------------
# 1. Cleaning
#    Goal: remove *formatting* differences between the two sources, so the
#    model has to learn writing style instead of dataset artifacts.
# --------------------------------------------------------------------------

# Human (Reddit/ELI5-style) answers in HC3 are PTB-tokenized:
#   "Do n't think of it ." / "( URL_0 )" / "it 's"
# ChatGPT answers are not. Left as-is, "space before period" alone is a
# near-perfect label leak.
_TOKENIZED_HINT = re.compile(r" [.,!?;:)\]] | n't\b| '(s|re|ve|ll|d|m)\b|\( |\[ ")

# ChatGPT answers lost their paragraph breaks: "...gland.The PSA test..."
# Humans almost never have that, so it is another leak.
_MISSING_SPACE = re.compile(r"([a-z0-9%)\]])([.!?:])([A-Z][a-z])")

_URL_MD = re.compile(r"\[\s*([^\]]+?)\s*\]\s*\(\s*URL_\d+\s*\)")  # [text](URL_0)
_URL_PAREN = re.compile(r"\(\s*URL_\d+\s*\)")
_URL_BARE = re.compile(r"\bURL_\d+\b")

_REFUSAL = re.compile(
    r"(?i)\bas an ai\b|\bi am an ai\b|\bi'm an ai\b|\bas a language model\b|"
    r"\bai language model\b|^i'?m sorry,? but\b|^i am sorry,? but\b"
)

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"',
                            "”": '"', "–": "-", "—": "-",
                            "…": "...", " ": " "})


def detokenize(text: str) -> str:
    toks = text.split(" ")
    # PTB-style straight quotes: alternate opening ``  / closing '' so the
    # Treebank detokenizer attaches them to the right word
    opened = False
    for i, t in enumerate(toks):
        if t == '"':
            toks[i] = "''" if opened else "``"
            opened = not opened
    text = _detok.detokenize(toks)
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)         # "well - defined"
    # things the Treebank detokenizer leaves behind
    text = re.sub(r"\b(do|does|did|is|are|was|were|has|have|had|ca|wo|could|"
                  r"would|should|ai|need|must|might) n't\b", r"\1n't", text,
                  flags=re.I)
    text = re.sub(r"\b(\w+) nt\b", r"\1nt", text)            # "do nt" -> "dont"
    text = re.sub(r" ([.,!?;:%])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    text = re.sub(r" ([)\]])", r"\1", text)
    text = re.sub(r"\.\s+\.\s+\.", "...", text)
    return text


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).translate(_QUOTE_MAP)
    text = re.sub(r"\s+", " ", text).strip()

    # links: "[a picture](URL_0)" -> "a picture"; "(URL_0)" -> ""; "URL_0" -> ""
    text = _URL_MD.sub(r"\1", text)
    text = _URL_PAREN.sub("", text)
    text = _URL_BARE.sub("", text)

    if _TOKENIZED_HINT.search(text):        # decided by the text, not the label
        text = detokenize(text)

    text = _MISSING_SPACE.sub(r"\1\2 \3", text)
    text = re.sub(r"([a-z]):(\d{1,2}\.) ", r"\1: \2 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r" ([.,!?;:])", r"\1", text)
    return text


# --------------------------------------------------------------------------
# 2. Near-duplicate grouping (so near-copies can't sit in both train and test)
# --------------------------------------------------------------------------

def near_duplicate_groups(texts, threshold=0.8, seed=0):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4),
                          min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    rows, cols = [], []
    for i in range(X.shape[0]):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if 1 - d >= threshold:
                rows.append(i)
                cols.append(j)
    from scipy.sparse import coo_matrix
    n = X.shape[0]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(A, directed=False)
    return labels


# --------------------------------------------------------------------------
# 3. Main pipeline
# --------------------------------------------------------------------------

def load_raw(raw_dir):
    frames = []
    for name in RAW_FILES:
        if raw_dir:
            path = os.path.join(raw_dir, name)
            frames.append(pd.read_json(path, lines=True))
        else:
            with urllib.request.urlopen(RAW_BASE + name) as r:
                lines = r.read().decode("utf-8").splitlines()
            frames.append(pd.DataFrame([json.loads(l) for l in lines if l]))
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"text": "Text", "label": "Label"})[["Label", "Text"]]


def group_split(df, groups, fracs=(0.7, 0.1, 0.2), seed=123):
    """Split by group id; each label is split separately so the
    proportions stay balanced in every split."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "validation": [], "test": []}
    for lab in sorted(df["Label"].unique()):
        sub_groups = np.unique(groups[df["Label"].values == lab])
        rng.shuffle(sub_groups)
        sizes = df[df["Label"] == lab].groupby(groups[df["Label"].values == lab]).size()
        sizes = sizes.loc[sub_groups].values
        cum = np.cumsum(sizes) / sizes.sum()
        cut1, cut2 = fracs[0], fracs[0] + fracs[1]
        assign = np.where(cum <= cut1, "train", np.where(cum <= cut2, "validation", "test"))
        g2split = dict(zip(sub_groups, assign))
        mask = df["Label"].values == lab
        for split in out:
            sel = np.array([g2split[g] == split for g in groups[mask]])
            out[split].append(df[mask][sel])
    return {k: pd.concat(v).sample(frac=1, random_state=seed).reset_index(drop=True)
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=None, help="local folder with the HC3-Plus en/*.jsonl")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--small_per_class", type=int, default=1000,
                    help="size of the quick-iteration subset (per class, train)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = []

    def note(msg):
        print(msg)
        log.append(msg)

    df = load_raw(args.raw_dir)
    note(f"raw rows: {len(df)}  label counts: {df.Label.value_counts().to_dict()}")

    df["Text"] = df["Text"].map(clean_text)

    n0 = len(df)
    df = df[~df["Text"].str.contains(_REFUSAL)]
    note(f"dropped AI refusals / 'as an AI' disclaimers: {n0 - len(df)}")

    n0 = len(df)
    df = df[df["Text"].str.split().str.len() >= args.min_words]
    note(f"dropped texts shorter than {args.min_words} words: {n0 - len(df)}")

    n0 = len(df)
    df = df.drop_duplicates(subset="Text")
    # a text that appears with *both* labels is ambiguous -> drop it entirely
    df = df[~df["Text"].duplicated(keep=False)]
    note(f"dropped exact duplicates: {n0 - len(df)}")
    df = df.reset_index(drop=True)

    groups = near_duplicate_groups(df["Text"].tolist(), seed=args.seed)
    n_multi = int((pd.Series(groups).value_counts() > 1).sum())
    note(f"near-duplicate clusters (cos >= 0.8, size > 1): {n_multi}")

    # balance: downsample human to the number of AI texts (whole groups kept
    # together because we sample groups, then trim)
    n_ai = int((df.Label == 1).sum())
    rng = np.random.default_rng(args.seed)
    human_idx = np.where(df.Label.values == 0)[0]
    human_groups = np.unique(groups[human_idx])
    rng.shuffle(human_groups)
    keep, count = [], 0
    g_sizes = pd.Series(groups[human_idx]).value_counts()
    for g in human_groups:
        if count >= n_ai:
            break
        keep.append(g)
        count += g_sizes[g]
    keep = set(keep)
    mask = (df.Label.values == 1) | np.array([g in keep for g in groups])
    df, groups = df[mask].reset_index(drop=True), groups[mask]
    note(f"after balancing: {df.Label.value_counts().to_dict()}")

    splits = group_split(df, groups, seed=args.seed)
    for name, part in splits.items():
        path = os.path.join(args.out_dir, f"{name}.csv")
        part[["Label", "Text"]].to_csv(path, index=None)
        note(f"{name}.csv: {len(part)} rows  {part.Label.value_counts().to_dict()}")

    tr = splits["train"]
    small = pd.concat([
        tr[tr.Label == lab].sample(min(int((tr.Label == lab).sum()), args.small_per_class),
                                   random_state=args.seed)
        for lab in (0, 1)
    ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    small.to_csv(os.path.join(args.out_dir, "train_small.csv"), index=None)
    note(f"train_small.csv: {len(small)} rows  {small.Label.value_counts().to_dict()}")

    with open(os.path.join(args.out_dir, "prep_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
