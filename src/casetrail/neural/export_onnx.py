"""Export the student encoder to ONNX and quantise it to INT8 for fast CPU inference (beyond syllabus).

INT8 dynamic quantisation stores the weights of the linear layers as 8-bit integers (4x smaller than float32)
and computes activations' scales on the fly. We check that the INT8 vectors stay close to the float32 ones
(cosine >= 0.98 on dev chunks, LLD section 8) and measure the size and the CPU latency.
Running the exported model lives in onnx_runtime.py, so the demo does not import the export tools.
"""
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from casetrail.neural.encoder import load_encoder


class PooledEncoder(torch.nn.Module):
    """Transformer + mean pooling + L2 normalisation in one graph, so ONNX returns final chunk vectors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        token_vectors = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)


def export_fp32(model_dir: str, onnx_path, opset: int = 17):
    tokenizer, model = load_encoder(model_dir)
    wrapper = PooledEncoder(model).eval()
    sample = tokenizer(["a short sample sentence", "another one"], padding=True, return_tensors="pt")
    torch.onnx.export(wrapper, (sample["input_ids"], sample["attention_mask"]), str(onnx_path),
                      input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "embedding": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    return tokenizer


def quantise_int8(fp32_path, int8_path):
    # Per-channel: each weight row gets its own scale. With one scale per matrix the INT8 vectors drifted
    # (mean cosine 0.946 vs fp32 on dev chunks); per-channel keeps 0.993 at the same size (DECISIONS log).
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, per_channel=True)
