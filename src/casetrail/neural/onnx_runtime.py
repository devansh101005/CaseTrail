"""Run an exported ONNX encoder on the CPU (used by the demo and by encode_corpus.py).

Only onnxruntime is imported here, not the export or quantisation tools: those pull in the `onnx` and
`ml_dtypes` packages, whose DLL Windows Smart App Control has been seen to block. The demo never needs them.
"""
import numpy as np
import onnxruntime as ort


def onnx_session(onnx_path, threads: int = 0) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def onnx_encode(session, tokenizer, texts: list, batch_size: int = 32, prefix: str = "",
                max_tokens: int = 256) -> np.ndarray:
    """Unit-length float32 vectors, one per text (the pooling and normalisation are inside the ONNX graph)."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start:start + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_tokens, return_tensors="np")
        feed = {"input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)}
        vectors.append(session.run(["embedding"], feed)[0])
    return np.concatenate(vectors).astype(np.float32)
