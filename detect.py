"""
Use the fine-tuned GPT-2 classifier: is a piece of English text human- or AI-written?

    python detect.py                    # paste text, then press Enter twice (Ctrl+C to quit)
    python detect.py --file essay.txt   # one text file
    python detect.py --file a.txt b.txt c.txt

How a text is scored
  * cleaned exactly like the training data (prepare_hape.clean_text)
  * cut into chunks of whole sentences, each up to 256 tokens (the model was
    trained on 256-token windows that start at a sentence boundary)
  * P(AI) for each chunk = softmax of the last token's logits; the text's
    score is the average over its chunks
  * verdict: P(AI) >= threshold -> "AI". Default threshold = the one that
    maximized accuracy on the validation set for this model.
    `python detect.py --calibrate` instead sets it so that only 5% of the
    human validation texts are flagged (fewer false accusations).

Limits (read these before trusting a verdict)
  * trained on HAP-E only: English, ~500-word passages, written by GPT-4o(-mini)
    and Llama-3 (8B/70B, base and instruct). Other / newer models were never
    seen, and heavily edited or paraphrased AI text is not covered.
  * texts shorter than ~150 words are outside what the model was trained on.

The first run downloads the trained weights (ai_text_classifier_run1.pth, ~500 MB)
from the GitHub release; after that everything runs offline except tiktoken's
one-time download of the GPT-2 vocabulary.
  * test-set results: AUC 0.97, but even at the 5%-false-positive setting
    about 1 in 20 human texts is flagged. Never use this alone to accuse anyone.
"""

import argparse
import json
import os
import re
import sys

import tiktoken
import torch

from previous_chapters import GPTModel
from prepare_hape import clean_text

# the trained weights are too big for git; they are attached to the GitHub release
RELEASE_MODEL = "ai_text_classifier_run1.pth"
RELEASE_URL = ("https://github.com/yangzihan0806-cpu/ai-text-detector/releases/download/"
               "v1.0/" + RELEASE_MODEL)


def ensure_model(path=RELEASE_MODEL, url=RELEASE_URL):
    """Download the trained weights from the GitHub release if they are not here yet."""
    if os.path.exists(path):
        return path
    import urllib.request
    print(f"{path} not found - downloading it once from the GitHub release (~500 MB) ...")
    tmp = path + ".part"

    last = [-1]

    def progress(blocks, block_size, total):
        done_mb = min(blocks * block_size, max(total, 0)) // 5_000_000 * 5
        if total > 0 and done_mb != last[0]:
            last[0] = done_mb
            print(f"\r  {done_mb:4d} / {total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    os.replace(tmp, path)
    print(f"\nSaved to {path}")
    return path

# accuracy-optimal thresholds found on the validation set by compare_models.py
DEFAULT_THRESHOLDS = {
    "ai_text_classifier_run1.pth": 0.5846,
    "ai_text_classifier.pth": 0.4163,
}
CHUNK_TOKENS = 256
MIN_LAST_CHUNK = 128          # a trailing chunk shorter than this is dropped (if there are others)
SHORT_TEXT_TOKENS = 200       # below this the result is flagged as unreliable


def load_model(path, device, tiny=False):
    if tiny:  # only for testing this script without the real GPT-2 weights
        cfg = {"vocab_size": 50257, "context_length": 256, "drop_rate": 0.0,
               "qkv_bias": False, "emb_dim": 12, "n_layers": 1, "n_heads": 2}
    else:
        cfg = {"vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0,
               "qkv_bias": True, "emb_dim": 768, "n_layers": 12, "n_heads": 12}
    model = GPTModel(cfg)
    model.out_head = torch.nn.Linear(cfg["emb_dim"], 2)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def sentence_chunks(text, tokenizer):
    """Pack whole sentences into chunks of <= CHUNK_TOKENS tokens."""
    # split after . ! ? (optionally followed by a closing quote/bracket) + whitespace
    sentences = [s for s in re.split(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])", text) if s]
    chunks, cur = [], []
    for s in sentences:
        ids = tokenizer.encode(s if not cur else " " + s)
        if cur and len(cur) + len(ids) > CHUNK_TOKENS:
            chunks.append(cur)
            cur = tokenizer.encode(s)
        else:
            cur = cur + ids
    if cur:
        chunks.append(cur)
    chunks = [c[:CHUNK_TOKENS] for c in chunks]          # a single very long sentence
    if len(chunks) > 1 and len(chunks[-1]) < MIN_LAST_CHUNK:
        chunks = chunks[:-1]
    return chunks


@torch.no_grad()
def score_text(text, model, tokenizer, device):
    text = clean_text(text)
    n_tokens = len(tokenizer.encode(text))
    chunks = sentence_chunks(text, tokenizer)
    probs = []
    for ids in chunks:  # one chunk at a time: no padding needed (training data had none)
        logits = model(torch.tensor(ids, device=device).unsqueeze(0))[:, -1, :]
        probs.append(torch.softmax(logits, dim=-1)[0, 1].item())
    return sum(probs) / len(probs), probs, n_tokens


def calibrate(model, tokenizer, device, model_file, fpr=0.05):
    import pandas as pd
    val = pd.read_csv("data/validation.csv")
    human = val[val.Label == 0].Text.tolist()
    print(f"Scoring {len(human)} human validation texts to set the threshold "
          f"(a few minutes on a CPU) ...")
    scores = []
    for i, t in enumerate(human, 1):
        ids = tokenizer.encode(t)[:CHUNK_TOKENS]   # same input the model saw in training
        with torch.no_grad():
            logits = model(torch.tensor(ids, device=device).unsqueeze(0))[:, -1, :]
        scores.append(torch.softmax(logits, dim=-1)[0, 1].item())
        if i % 200 == 0:
            print(f"  {i}/{len(human)}")
    scores.sort()
    thr = scores[int((1 - fpr) * len(scores))]
    with open("threshold.json", "w") as f:
        json.dump({"model": model_file, "threshold": thr, "rule": f"{fpr:.0%} FPR on validation"}, f)
    print(f"Threshold set to {thr:.4f} (flags {fpr:.0%} of human validation texts); saved to threshold.json")


def get_threshold(model_file, override):
    if override is not None:
        return override, "--threshold"
    if os.path.exists("threshold.json"):
        with open("threshold.json") as f:
            d = json.load(f)
        if d.get("model") == model_file:
            return d["threshold"], d["rule"]
    return DEFAULT_THRESHOLDS.get(model_file, 0.5), "accuracy-optimal on validation"


def report(text, model, tokenizer, device, threshold, name=None):
    p, chunk_probs, n_tokens = score_text(text, model, tokenizer, device)
    verdict = "AI-written" if p >= threshold else "human-written"
    if p >= 0.95 or p <= 0.05:
        strength = "confident"
    elif abs(p - threshold) < 0.15:
        strength = "uncertain"
    else:
        strength = "fairly confident"
    print("-" * 60)
    if name:
        print(name)
    print(f"P(AI) = {p:.3f}   ->  {verdict}  ({strength}; threshold {threshold:.3f})")
    if len(chunk_probs) > 1:
        print("per chunk: " + ", ".join(f"{q:.2f}" for q in chunk_probs))
    if n_tokens < SHORT_TEXT_TOKENS:
        print(f"WARNING: only {n_tokens} tokens (~{int(n_tokens * 0.75)} words). The model was "
              f"trained on ~500-word passages; results on short texts are unreliable.")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser(description="Human vs AI text detector (GPT-2 fine-tuned on HAP-E)")
    ap.add_argument("--model", default=None, help="path to the .pth (default: run1 if present)")
    ap.add_argument("--file", nargs="+", help="text file(s) to classify")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="set the threshold to 5%% false positives on the validation set")
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    model_file = args.model or next(
        (f for f in [RELEASE_MODEL, "ai_text_classifier.pth"] if os.path.exists(f)), None)
    if model_file is None:
        if args.tiny:
            sys.exit("No model found.")
        model_file = ensure_model()   # first run: fetch the trained weights from GitHub

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Loading {model_file} on {device} ...")
    model = load_model(model_file, device, tiny=args.tiny)

    if args.calibrate:
        calibrate(model, tokenizer, device, os.path.basename(model_file))
        return

    threshold, rule = get_threshold(os.path.basename(model_file), args.threshold)
    print(f"Threshold {threshold:.4f} ({rule})")

    if args.file:
        for path in args.file:
            with open(path, encoding="utf-8", errors="replace") as f:
                report(f.read(), model, tokenizer, device, threshold, name=path)
        return

    print("\nPaste a text, then press Enter twice (two empty lines end the text). Ctrl+C to quit.")
    while True:
        try:
            lines, empty = [], 0
            while True:
                line = input()
                if line.strip():
                    lines.append(line)
                    empty = 0
                elif lines:
                    empty += 1
                    if empty >= 2:
                        break
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return
        report(" ".join(lines), model, tokenizer, device, threshold)
        print("\nNext text (press Enter twice to finish it, Ctrl+C to quit):")


if __name__ == "__main__":
    main()
