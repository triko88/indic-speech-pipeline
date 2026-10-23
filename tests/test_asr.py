import unicodedata
from pathlib import Path
from types import SimpleNamespace

import onnxruntime
import pytest

from indic_s2t import asr
from indic_s2t.audio import load_16k_mono

ASR_DIR = Path(__file__).resolve().parent.parent / "data" / "asr"
SCRIPT_BLOCKS = {"hi": (0x0900, 0x097F), "ta": (0x0B80, 0x0BFF)}


class FakeSession:
    """Records what onnxruntime.InferenceSession would have been given."""

    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        self.path_or_bytes = path_or_bytes
        self.sess_options = sess_options
        self.providers = tuple(providers or ())

    def get_providers(self):
        return list(self.providers)

    def get_session_options(self):
        return self.sess_options


@pytest.fixture
def fake_onnxruntime(monkeypatch):
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    return FakeSession


def test_sessions_inside_context_get_threads_and_cpu_only(fake_onnxruntime):
    with asr.cpu_session_options(8):
        # same lookup the remote code does: `ort.InferenceSession(path, providers=[...])`
        created_session = onnxruntime.InferenceSession("encoder.onnx", providers=["CPUExecutionProvider"])
    assert isinstance(created_session, FakeSession)
    assert created_session.sess_options.intra_op_num_threads == 8
    assert created_session.sess_options.inter_op_num_threads == 1
    assert created_session.providers == ("CPUExecutionProvider",)


@pytest.mark.parametrize("requested_providers", [
    ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ["AzureExecutionProvider"],
    None,
])
def test_caller_providers_are_overridden_to_cpu(fake_onnxruntime, requested_providers):
    caller_options = onnxruntime.SessionOptions()
    caller_options.intra_op_num_threads = 2
    with asr.cpu_session_options(8):
        created_session = onnxruntime.InferenceSession("joint_enc.onnx", sess_options=caller_options,
                                                       providers=requested_providers)
    assert created_session.providers == ("CPUExecutionProvider",)
    assert created_session.sess_options.intra_op_num_threads == 8


def test_original_class_restored_after_normal_exit(fake_onnxruntime):
    with asr.cpu_session_options(8):
        assert onnxruntime.InferenceSession is not FakeSession
    assert onnxruntime.InferenceSession is FakeSession


def test_original_class_restored_after_exception(fake_onnxruntime):
    with pytest.raises(RuntimeError, match="model build failed"):
        with asr.cpu_session_options(8):
            raise RuntimeError("model build failed")
    assert onnxruntime.InferenceSession is FakeSession


def test_find_sessions_walks_object_graph(fake_onnxruntime):
    encoder_session = FakeSession("encoder.onnx", providers=["CPUExecutionProvider"])
    decoder_session = FakeSession("ctc_decoder.onnx", providers=["CPUExecutionProvider"])
    fake_model = SimpleNamespace(
        models={"encoder": encoder_session, "ctc_decoder": decoder_session, "missing": None, "preprocessor": object()},
        vocab={"hi": ["a", "b"]},
        alias_of_encoder=[encoder_session],
    )
    found_sessions = asr.find_sessions(fake_model, FakeSession)
    assert set(map(id, found_sessions)) == {id(encoder_session), id(decoder_session)}
    assert len(found_sessions) == 2


def test_summarize_sessions_reports_providers_and_threads(fake_onnxruntime):
    with asr.cpu_session_options(8):
        created_sessions = tuple(onnxruntime.InferenceSession(name) for name in ("encoder.onnx", "joint_enc.onnx"))
    asr_info = asr.summarize_sessions(created_sessions, injection="test")
    assert asr_info == asr.AsrInfo(threads=(8,), providers=("CPUExecutionProvider",), session_count=2,
                                   injection="test")


def script_fraction(transcript, lang):
    """Share of letters/marks in the transcript that fall in the language's Unicode block."""
    block_start, block_end = SCRIPT_BLOCKS[lang]
    script_chars = [char for char in transcript if unicodedata.category(char)[0] in "LM"]
    return sum(block_start <= ord(char) <= block_end for char in script_chars) / max(len(script_chars), 1)


@pytest.fixture(scope="module")
def loaded_asr():
    return asr.load_asr(threads=8)


@pytest.mark.hw
@pytest.mark.parametrize("lang", ["hi", "ta"])
def test_transcribe_real_wavs(loaded_asr, lang):
    wav_paths = sorted(set(ASR_DIR.glob(f"{lang}/*.wav")) | set(ASR_DIR.glob(f"{lang}_*.wav")))[:2]
    if len(wav_paths) < 2:
        pytest.skip(f"need 2 wavs in {ASR_DIR}/{lang}/ or {ASR_DIR}/{lang}_*.wav")
    model, asr_info = loaded_asr
    assert asr_info.providers == ("CPUExecutionProvider",)
    assert asr_info.threads == (8,)
    for wav_path in wav_paths:
        transcript = asr.transcribe(model, load_16k_mono(wav_path), lang)
        assert transcript.strip(), wav_path
        assert script_fraction(transcript, lang) > 0.8, (wav_path, transcript)
