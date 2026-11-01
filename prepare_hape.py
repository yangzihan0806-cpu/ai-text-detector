"""
Prepare HAP-E (Human-AI Parallel English Corpus) for the Chapter 6
classification fine-tuning in "Build a Large Language Model (From Scratch)".

HAP-E (Reinhart, Brown et al., CMU / NJIT; MIT license):
  https://huggingface.co/datasets/browndw/human-ai-parallel-corpus
  - 6 genres: acad, news, fic, spok, blog, tvm (COCA-like balance)
  - each human text's first ~1,000 words were split into two ~500-word chunks
      chunk_1 -> given to the LLM as the prompt ("continue in the same style")
      chunk_2 -> the real human continuation
  - 6 LLMs each wrote a continuation: gpt-4o, gpt-4o-mini, Llama-3-8B/70B
    (base) and Llama-3-8B/70B-Instruct
  - doc_id looks like "acad_0001@gpt-4o-2024-08-06" / "acad_0001@chunk_2"

What this script builds
  * human (Label 0) = chunk_2, AI (Label 1) = an LLM continuation of the same
    chunk_1  -> same topic, same genre, similar length, only the author differs
  * chunk_1 is dropped (it is the prompt, not a comparison text)
  * 1 human + 1 AI per document (AI model chosen round-robin so all 6 models
    are equally represented) -> perfectly balanced
  * split 70/10/20 BY DOCUMENT and stratified by genre, so a human text and
    the AI continuation of the same prompt never sit in different splits

Output: train.csv / validation.csv / test.csv / train_small.csv with columns
    Label, Text, Genre, Source
Label/Text are what the book's SpamDataset reads; Genre/Source are extra
columns (ignored by SpamDataset) for per-genre / per-model error analysis.

Usage
    pip install pandas datasets
    python prepare_hape.py --out_dir data
    python prepare_hape.py --input .  --out_dir data   # folder with the 8 downloaded parquet shards
"""

import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd

HF_NAME = "browndw/human-ai-parallel-corpus"

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"',
                            "”": '"', "–": "-", "—": "-",
                            "…": "...", " ": " "})

# COCA-derived human texts contain redaction runs like "@ @ @ @ @ @ @ @ @ @"
# (and sometimes "<p>" markup). LLM text has none -> pure label leak.
_COCA_REDACT = re.compile(r"(?:@\s*){2,}|\s@\s")
_HTML_TAG = re.compile(r"</?(p|br|h\d)\s*/?>", re.I)

# Llama-3 Instruct often prepends "Here is the continuation of the text in
# the same style and tone:" (~10% of its outputs) -> strip it
_LEAD_IN = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is| are)\b[^:\n]{0,150}?"
    r"(?:continu|next|complet|rest of|remain|style|tone|words)[^:\n]{0,100}:"
    r"|continuation\s*:)\s*", re.I)

_REFUSAL = re.compile(r"(?i)\bas an ai\b|\bai language model\b|"
                      r"^i'?m sorry,? but\b|^i cannot\b|^i can't\b")

# a sentence boundary: terminator (+ optional closing quote/paren) + space + a
# capital letter / opening quote
_SENT_END = re.compile(r"[.!?][\"')\]]?\s+(?=[\"'(\[]?[A-Z])")
_LAST_END = re.compile(r"^(.*[.!?][\"')\]]?)(?:\s|$)", re.S)


def trim_partial_sentences(text: str) -> str:
    """Human chunk_2 always starts and (almost always) ends at a sentence
    boundary, but LLM continuations often start mid-sentence ("...and we'd sit
    down", "which allowed us to") or get cut off at the token limit. Both are
    label leaks, so every text (whatever its label) is trimmed to whole
    sentences."""
    if re.match(r"^(\.\.\.|[a-z])", text):
        m = _SENT_END.search(text)
        text = text[m.end():] if m else text
    m = _LAST_END.match(text)
    return m.group(1) if m else text


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    # all dash styles ("--" in COCA, "—" in GPT-4o) -> one spaced hyphen
    text = re.sub(r"\s*(?:--|—|–)\s*", " - ", text)
    text = text.translate(_QUOTE_MAP)
    text = _HTML_TAG.sub(" ", text)
    text = _COCA_REDACT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEAD_IN.sub("", text)
    # Daily-Mail style orphan periods: "service, . protesters", "Obama's . speech"
    text = re.sub(r"([,;:])\s*\.\s", r"\1 ", text)
    text = re.sub(r"\s\.\s(?=[a-z])", " ", text)
    text = re.sub(r" ([.,!?;:])", r"\1", text)
    text = trim_partial_sentences(text)
    return text.strip()


def load(input_path):
    if input_path and os.path.isdir(input_path):        # folder of *.parquet shards
        import glob
        files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    elif input_path:
        if input_path.endswith(".parquet"):
            df = pd.read_parquet(input_path)
        elif input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        else:
            df = pd.read_json(input_path, lines=True)
    else:
        from datasets import load_dataset
        df = load_dataset(HF_NAME, split="train").to_pandas()
    return df[["doc_id", "text"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="local parquet/csv/jsonl copy of HAP-E")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--min_words", type=int, default=230,
                    help="texts must stay longer than the 256-token window, so padding can not leak length")
    ap.add_argument("--small_per_class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log = []

    def note(msg):
        print(msg)
        log.append(msg)

    df = load(args.input)
    parts = df["doc_id"].str.split("@", n=1, expand=True)
    df["Doc"], df["Source"] = parts[0], parts[1]
    df["Genre"] = df["Doc"].str.rsplit("_", n=1).str[0]
    note(f"raw rows: {len(df)}")
    note("sources: " + str(df["Source"].value_counts().to_dict()))
    note("genres: " + str(df["Genre"].value_counts().to_dict()))

    is_chunk = df["Source"].str.contains("chunk", case=False)
    human_src = [s for s in df.loc[is_chunk, "Source"].unique() if s.endswith("2")]
    assert len(human_src) == 1, f"could not find the human chunk_2 rows: {df.loc[is_chunk,'Source'].unique()}"
    human_src = human_src[0]

    df = df[(~is_chunk) | (df["Source"] == human_src)].copy()   # drop chunk_1 (the prompt)
    df["Label"] = (df["Source"] != human_src).astype(int)
    df["Text"] = df["text"].map(clean_text)

    n0 = len(df)
    df = df[~(df["Text"].str.contains(_REFUSAL) & (df["Label"] == 1))]
    df = df[df["Text"].str.split().str.len() >= args.min_words]
    df = df.drop_duplicates(subset="Text")
    note(f"dropped refusals / too short (<{args.min_words} words) / duplicates: {n0 - len(df)}")

    # keep docs that still have the human text and at least one AI text
    has_h = df[df.Label == 0].groupby("Doc").size()
    has_a = df[df.Label == 1].groupby("Doc").size()
    docs = np.array(sorted(set(has_h.index) & set(has_a.index)))
    note(f"usable documents (human + >=1 AI): {len(docs)}")

    # 1 human + 1 AI per doc; AI model picked round-robin (shuffled order)
    models = sorted(df.loc[df.Label == 1, "Source"].unique())
    rng.shuffle(docs)
    by_doc = {d: g for d, g in df[df.Doc.isin(set(docs))].groupby("Doc")}
    rows = []
    used = {m: 0 for m in models}
    for i, d in enumerate(docs):
        g = by_doc[d]
        rows.append(g[g.Label == 0].iloc[0])
        ai = g[g.Label == 1]
        avail = [m for m in models if m in set(ai.Source)]
        # the least-used available model (ties broken by a rotating order),
        # so every model ends up with ~the same number of texts
        pick = min(avail, key=lambda m: (used[m], (models.index(m) - i) % len(models)))
        used[pick] += 1
        rows.append(ai[ai.Source == pick].iloc[0])
    pairs = pd.DataFrame(rows).reset_index(drop=True)
    note(f"paired rows: {len(pairs)}  labels: {pairs.Label.value_counts().to_dict()}")
    note("AI models in pairs: " + str(pairs[pairs.Label == 1].Source.value_counts().to_dict()))

    # split by document, stratified by genre
    split_of = {}
    for genre, gdocs in pairs.groupby("Genre")["Doc"]:
        u = np.array(sorted(gdocs.unique()))
        rng.shuffle(u)
        n_tr, n_va = int(0.7 * len(u)), int(0.1 * len(u))
        for j, d in enumerate(u):
            split_of[d] = "train" if j < n_tr else ("validation" if j < n_tr + n_va else "test")
    pairs["Split"] = pairs["Doc"].map(split_of)

    cols = ["Label", "Text", "Genre", "Source"]
    for name in ["train", "validation", "test"]:
        part = pairs[pairs.Split == name].sample(frac=1, random_state=args.seed)
        part[cols].to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=None)
        note(f"{name}.csv: {len(part)} rows  {part.Label.value_counts().to_dict()}")
        if name == "train":
            # small subset: whole pairs, so it stays balanced
            small_docs = rng.choice(part.Doc.unique(),
                                    size=min(args.small_per_class, part.Doc.nunique()),
                                    replace=False)
            small = part[part.Doc.isin(set(small_docs))].sample(frac=1, random_state=args.seed)
            small[cols].to_csv(os.path.join(args.out_dir, "train_small.csv"), index=None)
            note(f"train_small.csv: {len(small)} rows  {small.Label.value_counts().to_dict()}")

    leak = set(pairs[pairs.Split == "train"].Doc) & set(pairs[pairs.Split == "test"].Doc)
    note(f"documents shared by train and test: {len(leak)}")

    w = pairs.Text.str.split().str.len()
    note("median words  human: %d  AI: %d" % (w[pairs.Label == 0].median(), w[pairs.Label == 1].median()))

    with open(os.path.join(args.out_dir, "prep_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
