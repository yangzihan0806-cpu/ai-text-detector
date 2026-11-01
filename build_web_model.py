"""
Build the model used by the browser version (docs/): PyTorch -> ONNX ->
8-bit weight-only quantization -> two files under 100 MB (GitHub's limit).

    pip install onnx onnxruntime
    python build_web_model.py                      # uses ai_text_classifier_run1.pth

Why weight-only: plain dynamic int8 quantization (which also quantizes the
activations) broke GPT-2 here: accuracy fell from 91% to 75%. Storing only the
weights in 8 bits (blocks of 32, MatMulNBits) and the token embedding in fp16,
while computing in fp32, keeps the predictions essentially unchanged
(420 HAP-E test texts: same accuracy, AUC 0.9668 vs 0.9671, 99.5% same verdicts).
"""
import os
import sys

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

from detect import CHUNK_TOKENS, load_model

src = sys.argv[1] if len(sys.argv) > 1 else "ai_text_classifier_run1.pth"
out_dir = os.path.join("docs", "model")
os.makedirs(out_dir, exist_ok=True)

# 1. export: input_ids (1, n) -> logits of the last token (1, 2)
model = load_model(src, "cpu")
for blk in model.trf_blocks:   # chunks are <= 256 tokens: a 256x256 causal mask is enough
    blk.att.mask = torch.triu(torch.ones(CHUNK_TOKENS, CHUNK_TOKENS), diagonal=1)


class LastTokenLogits(torch.nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, input_ids):
        return self.gpt(input_ids)[:, -1, :]


fp32_path = "model_fp32.onnx"
torch.onnx.export(LastTokenLogits(model).eval(), (torch.randint(0, 50257, (1, 100)),), fp32_path,
                  input_names=["input_ids"], output_names=["logits"],
                  dynamic_axes={"input_ids": {1: "seq"}}, opset_version=17, dynamo=False)

# 2. 8-bit weight-only quantization of every MatMul
q = MatMulNBitsQuantizer(onnx.load(fp32_path), bits=8, block_size=32, is_symmetric=True,
                         op_types_to_quantize=("MatMul",))
q.process()
m = q.model.model

# 3. token embedding (50257 x 768) stored in fp16, cast back to fp32 before the lookup
g = m.graph
inits = {i.name: i for i in g.initializer}
for node in g.node:
    if node.op_type == "Gather" and node.input[0] in inits:
        arr = numpy_helper.to_array(inits[node.input[0]])
        if arr.ndim == 2 and arr.shape[0] == 50257:
            new = numpy_helper.from_array(arr.astype(np.float16), node.input[0] + "_fp16")
            g.initializer.remove(inits[node.input[0]])
            g.initializer.append(new)
            cast_out = node.input[0] + "_as_fp32"
            g.node.insert(list(g.node).index(node),
                          helper.make_node("Cast", [new.name], [cast_out], to=TensorProto.FLOAT, name="emb_cast"))
            node.input[0] = cast_out
            break

data = m.SerializeToString()
os.remove(fp32_path)

# 4. split into two halves (< 100 MB each); the page joins them again
half = (len(data) + 1) // 2
for i, part in enumerate((data[:half], data[half:])):
    with open(os.path.join(out_dir, f"model.onnx.part{i}"), "wb") as f:
        f.write(part)
print(f"model: {len(data) / 1e6:.1f} MB in 2 parts -> {out_dir}  (update MODEL_BYTES in docs/index.html: {len(data)})")
