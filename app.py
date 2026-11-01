"""
Gradio web app: human vs AI text detector.
GPT-2 (124M) fine-tuned for classification (Chapter 6 of "Build a Large
Language Model (From Scratch)") on the HAP-E corpus.
"""

import os

import gradio as gr
import tiktoken
import torch

from detect import CHUNK_TOKENS, SHORT_TEXT_TOKENS, ensure_model, load_model, score_text

MODEL_FILE = "ai_text_classifier_run1.pth"   # downloaded from the GitHub release if missing
# threshold chosen on the HAP-E validation set so that only 5% of human texts are flagged
AI_THRESHOLD = 0.817
HUMAN_THRESHOLD = 0.5
MAX_CHARS = 20000

TINY = os.environ.get("TINY_TEST") == "1"   # only for testing without the real weights

torch.set_num_threads(max(1, os.cpu_count() or 1))
device = torch.device("cpu")
tokenizer = tiktoken.get_encoding("gpt2")
model = load_model(MODEL_FILE if TINY else ensure_model(MODEL_FILE), device, tiny=TINY)


def classify(text):
    text = (text or "").strip()
    if len(text.split()) < 5:
        return "", "Please paste some English text (ideally 150+ words)."
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    p, chunk_probs, n_tokens = score_text(text, model, tokenizer, device)

    if p >= AI_THRESHOLD:
        verdict, color = "Likely AI-generated", "#d9480f"
    elif p < HUMAN_THRESHOLD:
        verdict, color = "Likely human-written", "#2b8a3e"
    else:
        verdict, color = "Uncertain", "#868e96"

    lines = [f"*Likely AI* if P(AI) ≥ {AI_THRESHOLD}, *likely human* if below "
             f"{HUMAN_THRESHOLD}, otherwise *uncertain*."]
    if len(chunk_probs) > 1:
        lines.append(f"The text was split into {len(chunk_probs)} chunks of up to "
                     f"{CHUNK_TOKENS} tokens; P(AI) per chunk: "
                     + ", ".join(f"{q:.2f}" for q in chunk_probs))
    if n_tokens < SHORT_TEXT_TOKENS:
        lines.append(f"⚠️ Only about {int(n_tokens * 0.75)} words. The model was trained on "
                     f"~500-word passages, so results for short texts are unreliable.")
    return verdict_html(verdict, color, p), "\n\n".join(lines)


def verdict_html(verdict, color, p):
    def mark(x, label):
        return (f'<div style="position:absolute;left:{x*100:.1f}%;top:-4px;bottom:-4px;'
                f'border-left:2px dashed currentColor;opacity:.55"></div>'
                f'<div style="position:absolute;left:{x*100:.1f}%;top:22px;transform:translateX(-50%);'
                f'font-size:12px;opacity:.7">{label}</div>')
    return f"""
<div style="padding:4px 2px 30px 2px">
  <div style="font-size:26px;font-weight:700;color:{color};margin-bottom:6px">{verdict}</div>
  <div style="font-size:16px;margin-bottom:12px">P(AI) = <b>{p:.3f}</b></div>
  <div style="position:relative;height:16px;border-radius:8px;background:rgba(128,128,128,.25)">
    <div style="width:{p*100:.1f}%;height:100%;border-radius:8px;background:{color}"></div>
    {mark(HUMAN_THRESHOLD, "0.5")}{mark(AI_THRESHOLD, str(AI_THRESHOLD))}
  </div>
</div>"""


ABOUT = """
**How it works.** GPT-2 small (124M parameters), pretrained by OpenAI, fine-tuned for
binary classification following Chapter 6 of *Build a Large Language Model (From Scratch)*
(S. Raschka): the last transformer block, the final LayerNorm and a new 2-class output head
were trained on the [HAP-E corpus](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus)
(human passages vs. continuations of the same passages by GPT-4o, GPT-4o-mini and Llama-3 8B/70B,
base and instruct). Long texts are split into sentence-aligned chunks of up to 256 tokens and the
chunk scores are averaged.

**Results.** HAP-E test set: ROC AUC 0.971 (TF-IDF baseline 0.947); about 67-77% of Llama-3
*base*-model texts are caught at a 5% false-positive rate, 95%+ for the chat/instruct models.
Unseen data set (HC3, 2022 ChatGPT answers): AUC 0.988, but 10.6% of human answers were
flagged at the threshold used here.

**Limits.** English only. Trained on ~500-word passages; short texts are unreliable. Never saw
newer models (e.g. GPT-5, Claude, Gemini) or human-edited / paraphrased AI text. Some human
texts *will* be flagged. **Do not use this tool on its own to accuse anyone of using AI.**
"""

with gr.Blocks(title="AI Text Detector (GPT-2)") as demo:
    gr.Markdown("# AI Text Detector\nPaste an English text (150+ words works best).")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="Text", lines=14, max_lines=30,
                             placeholder="Paste text here ...")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            out_label = gr.HTML()
            out_md = gr.Markdown()
    with gr.Accordion("About this model, results and limits", open=False):
        gr.Markdown(ABOUT)
    btn.click(classify, inputs=inp, outputs=[out_label, out_md])
    inp.submit(classify, inputs=inp, outputs=[out_label, out_md])

if __name__ == "__main__":
    demo.launch()
