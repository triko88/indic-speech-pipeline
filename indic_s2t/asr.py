"""IndicConformer-600M ASR (hi, ta) on CPU via its trust_remote_code ONNX wrapper."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from indic_s2t.audio import load_16k_mono

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
# Real files, not the HF cache: onnxruntime >= 1.2x rejects external-data files that resolve (via the
# cache's blob symlinks) outside the model directory.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "indic-conformer-600m"
CPU_PROVIDERS = ("CPUExecutionProvider",)
# The remote model_onnx.py does `import onnxruntime as ort` and calls `ort.InferenceSession(...)` at
# construction time with no sess_options, so swapping the module attribute reaches every session.
INJECTION = "scoped patch of onnxruntime.InferenceSession (sess_options + CPU providers)"


@dataclass(frozen=True)
class AsrInfo:
    threads: tuple[int, ...]  # distinct intra_op_num_threads observed on the loaded sessions
    providers: tuple[str, ...]  # distinct providers across sessions, in first-seen order
    session_count: int
    injection: str


def cpu_threaded_options(threads):
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1
    return session_options


@contextmanager
def cpu_session_options(threads):
    """Every InferenceSession built inside gets fresh CPU-only options with `threads` intra-op threads.

    Caller-supplied sess_options/providers/provider_options are replaced, not merged."""
    original_session_class = onnxruntime.InferenceSession

    class CpuInferenceSession(original_session_class):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            super().__init__(path_or_bytes, sess_options=cpu_threaded_options(threads),
                             providers=list(CPU_PROVIDERS), **kwargs)

    onnxruntime.InferenceSession = CpuInferenceSession
    try:
        yield
    finally:
        onnxruntime.InferenceSession = original_session_class


def find_sessions(root, session_class, max_depth=3):
    """Sessions reachable from `root` through attributes, dict values and list/tuple items."""
    if isinstance(root, session_class):
        return (root,)
    if max_depth == 0:
        return ()
    if isinstance(root, dict):
        children = tuple(root.values())
    elif isinstance(root, (list, tuple)):
        children = tuple(root)
    elif hasattr(root, "__dict__"):
        children = tuple(vars(root).values())
    else:
        return ()
    found_sessions = (session for child in children for session in find_sessions(child, session_class, max_depth - 1))
    return tuple({id(session): session for session in found_sessions}.values())


def summarize_sessions(sessions, injection):
    return AsrInfo(
        threads=tuple(dict.fromkeys(session.get_session_options().intra_op_num_threads for session in sessions)),
        providers=tuple(dict.fromkeys(provider for session in sessions for provider in session.get_providers())),
        session_count=len(sessions),
        injection=injection,
    )


def load_asr(threads=8, model_id=MODEL_ID, model_dir=MODEL_DIR):
    torch.set_num_threads(threads)  # the mel preprocessor is TorchScript, outside ONNX Runtime's thread pool
    local_dir = snapshot_download(model_id, local_dir=model_dir)
    # Same as the remote IndicASRModel.from_pretrained (snapshot_download + constructor), minus the cache path.
    model_class = get_class_from_dynamic_module("model_onnx.IndicASRModel", local_dir)
    with cpu_session_options(threads):
        model = model_class(model_class.config_class(ts_folder=local_dir))
    return model, summarize_sessions(find_sessions(model, onnxruntime.InferenceSession), INJECTION)


def transcribe(model, wave_16k: np.ndarray, lang: str, decoding="rnnt") -> str:
    if wave_16k.ndim != 1:
        raise ValueError(f"expected mono 1-D wave, got shape {wave_16k.shape}")
    wave_tensor = torch.from_numpy(np.ascontiguousarray(wave_16k, dtype=np.float32)).unsqueeze(0)  # [1, T]
    with torch.inference_mode():
        return model(wave_tensor, lang, decoding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--decoding", default="rnnt", choices=("rnnt", "ctc"))
    parser.add_argument("--threads", type=int, default=8)
    cli_args = parser.parse_args()
    model, asr_info = load_asr(cli_args.threads)
    print(asr_info)
    print(transcribe(model, load_16k_mono(cli_args.wav), cli_args.lang, cli_args.decoding))


if __name__ == "__main__":
    main()
