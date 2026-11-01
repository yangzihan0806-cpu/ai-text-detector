"""
Cross-dataset test: the classifier was fine-tuned on HAP-E (GPT-4o / Llama-3
continuations of ~500-word passages). Here it is applied, unchanged, to the
HC3 test set (2022 ChatGPT vs human answers to questions; Label 0 = human,
1 = ChatGPT), which it has never seen.

Each text is scored exactly like detect.py does it (same cleaning, whole-
sentence chunks of <= 256 tokens, no padding, P(AI) averaged over chunks).

    python eval_hc3.py                       # uses ai_text_classifier_run1.pth
    python eval_hc3.py --model ai_text_classifier.pth

Reports AUC, accuracy at the HAP-E thresholds (accuracy-optimal and 5% FPR,
both chosen on the HAP-E validation set), a breakdown by text length, and the
best threshold on HC3 itself (optimistic: chosen on the test set, shown only to
separate "can it rank?" from "is the threshold right?").
"""

import argparse
import os

import numpy as np
import pandas as pd
import tiktoken
import torch
from sklearn.metrics import roc_auc_score

from detect import DEFAULT_THRESHOLDS, load_model, score_text


@torch.no_grad()
def hape_fpr_threshold(model, tokenizer, device, fpr=0.05):
    val = pd.read_csv("data/validation.csv")
    scores = []
    for t in val[val.Label == 0].Text:
        ids = tokenizer.encode(t)[:256]
        logits = model(torch.tensor(ids, device=device).unsqueeze(0))[:, -1, :]
        scores.append(torch.softmax(logits, dim=-1)[0, 1].item())
    return float(np.quantile(scores, 1 - fpr))


def summarize(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"acc": (pred == y).mean(),
            "human acc (1-FPR)": (pred[y == 0] == 0).mean(),
            "AI acc (TPR)": (pred[y == 1] == 1).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai_text_classifier_run1.pth")
    ap.add_argument("--data", default="data/hc3/test.csv")
    ap.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = tiktoken.get_encoding("gpt2")
    print(f"Device: {device}   model: {args.model}   data: {args.data}")
    model = load_model(args.model, device, tiny=args.tiny)

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.groupby("Label", group_keys=False).head(args.limit // 2).reset_index(drop=True)
    print(f"Scoring {len(df)} HC3 texts ...")
    probs, ntok = [], []
    for i, t in enumerate(df.Text, 1):
        p, _, n = score_text(t, model, tok, device)
        probs.append(p)
        ntok.append(n)
        if i % 500 == 0:
            print(f"  {i}/{len(df)}")
    df["p_ai"], df["n_tokens"] = probs, ntok
    y, p = df.Label.values, df.p_ai.values

    thr_acc = DEFAULT_THRESHOLDS.get(os.path.basename(args.model), 0.5)
    print("Computing the 5%-FPR threshold on the HAP-E validation set ...")
    thr_fpr = hape_fpr_threshold(model, tok, device)
    cand = np.unique(p)
    thr_oracle = float(cand[np.argmax([((p >= c) == y).mean() for c in cand])])

    print("\n" + "=" * 70)
    print(f"HC3 test  (n={len(df)}: {int((y == 0).sum())} human, {int((y == 1).sum())} ChatGPT)")
    print(f"AUC: {roc_auc_score(y, p):.4f}\n")
    rows = {f"threshold 0.5": summarize(y, p, 0.5),
            f"HAP-E acc-optimal ({thr_acc:.3f})": summarize(y, p, thr_acc),
            f"HAP-E 5% FPR ({thr_fpr:.3f})": summarize(y, p, thr_fpr),
            f"best on HC3 itself ({thr_oracle:.3f}) [optimistic]": summarize(y, p, thr_oracle)}
    print(pd.DataFrame(rows).T.round(4).to_string())

    print("\nBy text length (tokens), at the HAP-E acc-optimal threshold:")
    df["length"] = pd.cut(df.n_tokens, [0, 100, 200, 100000], labels=["<100", "100-200", ">=200"])
    out = []
    for b, g in df.groupby("length", observed=True):
        auc = roc_auc_score(g.Label, g.p_ai) if g.Label.nunique() == 2 else float("nan")
        s = summarize(g.Label.values, g.p_ai.values, thr_acc)
        out.append({"length": b, "n human": int((g.Label == 0).sum()), "n AI": int((g.Label == 1).sum()),
                    "AUC": auc, **s})
    print(pd.DataFrame(out).set_index("length").round(4).to_string())

    print("\nMean P(AI):  human %.3f   ChatGPT %.3f" % (p[y == 0].mean(), p[y == 1].mean()))
    name = os.path.splitext(os.path.basename(args.model))[0]
    df[["Label", "n_tokens", "p_ai", "Text"]].to_csv(f"hc3_preds_{name}.csv", index=False)
    print(f"Per-text predictions saved to hc3_preds_{name}.csv")


if __name__ == "__main__":
    main()
