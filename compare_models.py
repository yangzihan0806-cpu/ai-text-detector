"""
Compare fine-tuned classifiers fairly, independent of the 0.5 decision threshold.

For every model (.pth saved by main.py) this script
  1. computes P(AI) = softmax(logits of the last token)[1] for every
     validation and test text,
  2. reports ROC AUC (threshold-free: how well the model *ranks* AI texts
     above human texts), overall and per generating model,
  3. reports accuracy at the default threshold 0.5 and at the threshold that
     maximizes accuracy on the VALIDATION set (chosen on val, scored on test),
  4. reports the detection rate per AI model at a fixed 5% false-positive
     rate (threshold chosen on val so that 5% of human val texts are flagged),
  5. runs a paired bootstrap on the test set for the AUC difference between
     the first two models, overall and for the two Llama-3 base models.
Predictions are saved to preds_<model>.csv for further analysis.

Usage (in the hape_ai_detect folder, on Colab with a GPU):
    python compare_models.py
    python compare_models.py --models ai_text_classifier_run1.pth ai_text_classifier.pth
"""

import argparse
import os

import numpy as np
import pandas as pd
import tiktoken
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from main import SpamDataset
from previous_chapters import GPTModel

HUMAN = "chunk_2"


def build_model(path, tiny=False):
    if tiny:  # only for testing this script without the real GPT-2
        cfg = {"vocab_size": 50257, "context_length": 256, "drop_rate": 0.0,
               "qkv_bias": False, "emb_dim": 12, "n_layers": 1, "n_heads": 2}
    else:     # same as main.py: gpt2-small (124M)
        cfg = {"vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0,
               "qkv_bias": True, "emb_dim": 768, "n_layers": 12, "n_heads": 12}
    model = GPTModel(cfg)
    model.out_head = torch.nn.Linear(cfg["emb_dim"], 2)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


@torch.no_grad()
def predict_proba(model, dataset, device, batch_size=32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = []
    for x, _ in loader:
        logits = model(x.to(device))[:, -1, :]
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def best_threshold(y, p):
    """Threshold that maximizes accuracy (searched over all distinct scores)."""
    cand = np.unique(np.concatenate([p, [0.5]]))
    accs = [((p >= t).astype(int) == y).mean() for t in cand]
    return float(cand[int(np.argmax(accs))])


def fpr_threshold(y, p, fpr=0.05):
    """Threshold so that a fraction `fpr` of the human texts score above it."""
    return float(np.quantile(p[y == 0], 1 - fpr))


def auc_by_source(df, col):
    out = {}
    human = df[df.Source == HUMAN]
    for src in sorted(s for s in df.Source.unique() if s != HUMAN):
        sub = pd.concat([human, df[df.Source == src]])
        out[src] = roc_auc_score(sub.Label, sub[col])
    return out


def paired_bootstrap_auc(df, col_a, col_b, sources=None, n_boot=1000, seed=0):
    """95% CI of AUC(b) - AUC(a), resampling human and AI texts separately."""
    rng = np.random.default_rng(seed)
    human = df[df.Source == HUMAN]
    ai = df[df.Label == 1] if sources is None else df[df.Source.isin(sources)]
    diffs = []
    for _ in range(n_boot):
        h = human.sample(len(human), replace=True, random_state=int(rng.integers(1e9)))
        a = ai.sample(len(ai), replace=True, random_state=int(rng.integers(1e9)))
        s = pd.concat([h, a])
        diffs.append(roc_auc_score(s.Label, s[col_b]) - roc_auc_score(s.Label, s[col_a]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["ai_text_classifier_run1.pth", "ai_text_classifier.pth"])
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tok = tiktoken.get_encoding("gpt2")
    val = SpamDataset("data/validation.csv", tok, max_length=args.max_length)
    test = SpamDataset("data/test.csv", tok, max_length=args.max_length)
    vdf, tdf = val.data.copy(), test.data.copy()

    names = []
    for path in args.models:
        name = os.path.splitext(os.path.basename(path))[0]
        names.append(name)
        print(f"\nScoring {path} ...")
        model = build_model(path, tiny=args.tiny).to(device)
        vdf[name] = predict_proba(model, val, device)
        tdf[name] = predict_proba(model, test, device)
        tdf[["Label", "Genre", "Source", name]].to_csv(f"preds_{name}.csv", index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    det_rows = {}
    for n in names:
        yv, pv, yt, pt = vdf.Label.values, vdf[n].values, tdf.Label.values, tdf[n].values
        t_acc = best_threshold(yv, pv)
        t_fpr = fpr_threshold(yv, pv, 0.05)
        rows.append({
            "model": n,
            "test AUC": roc_auc_score(yt, pt),
            "acc @0.5": ((pt >= 0.5) == yt).mean(),
            "tuned threshold (val)": t_acc,
            "acc @tuned": ((pt >= t_acc) == yt).mean(),
            "human acc @tuned": ((pt[yt == 0] >= t_acc) == 0).mean(),
            "test FPR @5%-val-threshold": (pt[yt == 0] >= t_fpr).mean(),
        })
        aucs = auc_by_source(tdf, n)
        det = tdf[tdf.Label == 1].assign(hit=tdf[n] >= t_fpr).groupby("Source").hit.mean()
        det_rows[n] = pd.DataFrame({"AUC (vs human)": aucs, "detected @5% FPR": det})

    pd.set_option("display.width", 140)
    print("\n" + "=" * 70 + "\nOverall (test set)\n")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())

    for n in names:
        print("\n" + "-" * 70 + f"\nPer AI model: {n}\n")
        print(det_rows[n].round(4).to_string())

    if len(names) >= 2:
        a, b = names[0], names[1]
        print("\n" + "=" * 70 + f"\nPaired bootstrap, AUC({b}) - AUC({a}), {args.n_boot} resamples\n")
        for label, srcs in [("all AI texts", None),
                            ("Meta-Llama-3-8B", ["Meta-Llama-3-8B"]),
                            ("Meta-Llama-3-70B", ["Meta-Llama-3-70B"])]:
            m, lo, hi = paired_bootstrap_auc(tdf, a, b, srcs, n_boot=args.n_boot)
            verdict = "significant" if (lo > 0 or hi < 0) else "not significant (CI contains 0)"
            print(f"{label:18s} diff {m:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")


if __name__ == "__main__":
    main()
