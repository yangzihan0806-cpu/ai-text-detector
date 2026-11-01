# AI Text Detector: GPT-2 fine-tuned to tell human from AI-written text

A small, fully reproducible project built on **Chapter 6 ("Fine-tuning for classification")** of
*Build a Large Language Model (From Scratch)* by Sebastian Raschka. The chapter's spam classifier
is turned into an **AI-generated-text detector**: the model and training loop stay exactly as in the
book (pretrained GPT-2 small, all layers frozen except the last transformer block, the final
LayerNorm and a new 2-class output head); only the data changes.

**Try it in your browser:** https://yangzihan0806-cpu.github.io/ai-text-detector/
(the model runs locally in the page; pasted text is never uploaded)

**Headline results**

| | HAP-E test (in-domain) | HC3 test (never seen) |
|---|---|---|
| TF-IDF + logistic regression (baseline) | ROC AUC 0.947 | ROC AUC **0.625** |
| **GPT-2 fine-tuned (this repo)** | ROC AUC **0.971** | ROC AUC **0.988** |

The bag-of-words baseline mostly learns which words GPT-4o / Llama-3 like and collapses on a new
data set; the fine-tuned GPT-2 transfers to 2022 ChatGPT answers it never saw.

---

## Data

**Training: [HAP-E](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus)**
(Human-AI Parallel English Corpus, Reinhart, Brown et al., CMU/NJIT, MIT license). Each human text is cut
into two ~500-word halves; six LLMs (GPT-4o, GPT-4o-mini, Llama-3-8B/70B base and instruct) continue the
first half, and the real second half is the human sample. Topic, genre and length are matched, so only
the author differs.

`prepare_hape.py` builds `data/`:
- 1 human + 1 AI text per document (8,276 pairs; the AI model is picked so all six are equally
  represented), split **by document** 70/10/20 and stratified by genre:
  `train.csv.gz` (11,580), `validation.csv` (1,650), `test.csv` (3,322), `train_small.csv` (2,000).
- Columns `Label, Text, Genre, Source` (Label 0 = human, 1 = AI).
- **Label-leak cleaning.** Formatting differences that give the label away were removed from all
  texts regardless of label: COCA `--` vs GPT-4o `—` dashes, stray ` .` in news text, "Here is the
  continuation:" lead-ins, LLM continuations starting mid-sentence or cut off at the token limit,
  and too-short texts (every text fills the 256-token window, so padding cannot leak length).
  Before cleaning, some of these features differed by up to ~23 percentage points between the classes;
  after cleaning, none differs by more than 0.1 points.

**Cross-data-set test: [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)** (human vs 2022
ChatGPT answers, CC-BY-SA). `prepare_hc3.py` builds `data/hc3/` (balanced, PTB-detokenized human answers,
URL placeholders and ChatGPT's missing paragraph spaces fixed, near-duplicates kept in the same split).

## Model and training (`main.py`)

Exactly the book's Chapter 6 setup: GPT-2 small (124M) with OpenAI's weights, `max_length=256`,
AdamW (lr 5e-5, weight decay 0.1), batch size 8, up to 5 epochs, loss = cross-entropy on the last
token's logits. Trainable: last transformer block + final LayerNorm + new `Linear(768, 2)` head
(≈7.09M of 124M parameters).

Additions to the book's script: early stopping on the full validation loss (best epoch saved),
per-model and per-genre test accuracy, a run log, `--eval_only`, and `classify_text` without padding
(training texts never contained padding, so a padded last position would be meaningless).

Training on a Colab T4 GPU took about 25 minutes (5 epochs).

## Results

### In-domain (HAP-E test, 3,322 texts)

Threshold-free comparison (`compare_models.py`): ROC AUC per generating model (vs all human texts) and the
share of each model's texts detected when the threshold is set on the validation set so that only 5%
of human texts are flagged.

| AI source | TF-IDF AUC | GPT-2 AUC | TF-IDF detected @5% FPR | GPT-2 detected @5% FPR |
|---|---|---|---|---|
| Llama-3-70B (base) | 0.830 | **0.915** | 44.3% | **66.6%** |
| Llama-3-8B (base) | 0.885 | **0.950** | 51.0% | **76.7%** |
| Llama-3-70B-Instruct | 0.984 | **0.991** | 93.0% | **96.3%** |
| Llama-3-8B-Instruct | 0.992 | 0.992 | 94.8% | **97.2%** |
| GPT-4o | **0.994** | 0.989 | **96.8%** | 96.1% |
| GPT-4o-mini | **0.993** | 0.986 | **97.5%** | 95.4% |
| **all** | 0.947 | **0.971** | | |

- The hard cases are the **base** (non-chat) models, whose output reads like human text; that is where
  GPT-2 beats the bag-of-words baseline by 22-26 points. Chat models are easy for both.
- Accuracy at threshold 0.5: 91.2%. Early stopping (best epoch 3 of 5) gave the same AUC
  (0.970 vs 0.971; paired bootstrap 95% CI of the difference contains 0). It only moved the
  decision threshold, i.e. traded human-text accuracy against base-model recall.

### Cross-data-set (HC3 test, 3,932 texts, model unchanged; `eval_hc3.py`)

| | value |
|---|---|
| ROC AUC | **0.988** (TF-IDF trained on HAP-E: 0.625) |
| accuracy at the HAP-E 5%-FPR threshold (0.817) | 93.4% |
| human texts flagged at that threshold | **10.6%** (target was 5%) |
| ChatGPT texts detected | 97.4% |
| AUC on texts < 100 tokens | 0.930 |

The ranking transfers, the probability scale does not: thresholds must be re-calibrated on the target
domain, and short texts are much less reliable.

## Web version (`docs/`, GitHub Pages)

The detector also runs entirely in the browser, so it can be hosted as a static page with no server.

- `build_web_model.py` exports the fine-tuned model to ONNX (input: token ids, output: last-token
  logits), stores the MatMul weights in 8 bits (weight-only, blocks of 32) and the token embedding in
  fp16, and splits the 177 MB result into two files under GitHub's 100 MB limit.
  Plain dynamic int8 quantization was tried first and broke the model (accuracy 91% → 75%);
  weight-only 8-bit keeps it intact: on 420 HAP-E test texts the accuracy is unchanged (90.95%),
  AUC 0.9668 vs 0.9671, and 99.5% of verdicts at the 0.817 threshold are the same.
- `docs/detector.js` is a line-by-line port of `clean_text` and `sentence_chunks`; on 1,804 raw texts it
  produces exactly the same cleaned text and token chunks as Python. The GPT-2 tokenizer
  ([js-tiktoken](https://github.com/dqbd/tiktoken), bundled with esbuild from `tools/tokenizer_entry.js`)
  matched Python tiktoken on 226,029 tokens with no differences.
- `docs/index.html` runs the model with [onnxruntime-web](https://onnxruntime.ai/) (WebAssembly).
  In Chromium its P(AI) matched the Python ONNX pipeline to within 5e-7 and the original PyTorch model to
  within 0.021 on 27 test texts, with no change of verdict. The model is downloaded once (177 MB)
  and kept in the browser's cache.

## Limits

English only; trained on ~500-word passages; never saw newer models (GPT-6, Claude Fable, ...) or
human-edited / paraphrased AI text, especially well-prompted text after human revision; on HC3 about 1 in 10 human answers was flagged.
Generally works best when recognizing GPT-based content.
**Do not use this detector on its own to accuse anyone of using AI.**

## How to run

```bash
pip install -r requirements.txt

# train (downloads GPT-2 124M on first run; a GPU is strongly recommended)
python main.py --test_mode     # seconds: checks the code with a tiny random model
python main.py --small         # 2,000 training texts
python main.py                 # full training set -> ai_text_classifier.pth

# evaluate
python compare_models.py --models ai_text_classifier.pth
python eval_hc3.py --model ai_text_classifier.pth

# just use the trained detector (no training needed; downloads the weights on first run)
python detect.py --file essay.txt
python detect.py               # paste text, then press Enter twice

# or use a model you trained yourself
python detect.py --model ai_text_classifier.pth --file essay.txt
python detect.py --calibrate   # set the threshold to 5% false positives on validation
python app.py                  # local web app (Gradio)
```

**Trained weights:** `ai_text_classifier_run1.pth` (500 MB) is too large for git, so it is attached
to the [v1.0 release](../../releases/tag/v1.0). `detect.py` and `app.py` download it automatically
on first run; you can also download it by hand and put it next to `detect.py`.

**Colab:** `notebooks/run_on_colab.ipynb` clones this repo and trains on a free T4 GPU.

**Rebuilding the data** (not needed, the prepared CSVs are in `data/`):
```bash
python prepare_hape.py --out_dir data            # downloads HAP-E from Hugging Face
python prepare_hc3.py --out_dir data/hc3         # downloads the HC3 QA files from GitHub
```

## Files

| file | purpose |
|---|---|
| `main.py` | Chapter 6 fine-tuning (book code + early stopping, logging, per-group accuracy) |
| `previous_chapters.py`, `gpt_download.py` | GPT model and OpenAI weight loader from the book's repo |
| `prepare_hape.py`, `prepare_hc3.py` | data download, cleaning, balancing, splitting |
| `compare_models.py` | AUC, tuned thresholds, detection @5% FPR, paired bootstrap |
| `eval_hc3.py` | cross-data-set test on HC3 |
| `detect.py` | command-line detector (sentence-aligned 256-token chunks, averaged P(AI)) |
| `app.py` | Gradio web app (needs a Python server) |
| `build_web_model.py`, `docs/` | browser version for GitHub Pages |

## Credits

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024; code from
  [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0).
- HAP-E: Reinhart, Brown et al., [browndw/human-ai-parallel-corpus](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus) (MIT).
- HC3: Guo et al. 2023, [arXiv:2301.07597](https://arxiv.org/abs/2301.07597) (CC-BY-SA 4.0); HC3 QA files
  re-hosted by [HC3-Plus](https://github.com/suu990901/chatgpt-comparison-detection-HC3-Plus).
- GPT-2 weights: OpenAI (MIT).
