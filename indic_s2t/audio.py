"""16 kHz mono loading and deterministic degradation conditions (babble, telephony, white noise).

Module-level imports stay numpy/scipy/soundfile only: the TTS env imports this too.
"""

import hashlib
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import butter, resample_poly, sosfiltfilt

SAMPLE_RATE = 16000
TELEPHONY_RATE = 8000
BABBLE_TALKERS = 5
MU = 255
STEP_PATTERN = re.compile(r"(babble|white)_(-?\d+(?:\.\d+)?)db|clean|telephony")


@dataclass(frozen=True)
class BabbleSource:
    utt_id: str
    speaker_id: str
    wave: np.ndarray


def resample(wave, source_rate, target_rate):
    if source_rate == target_rate:
        return wave
    common_divisor = math.gcd(source_rate, target_rate)
    return resample_poly(wave, target_rate // common_divisor, source_rate // common_divisor)


def decode_with_ffmpeg(path):
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "decoded.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                        "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav_path)], check=True)
        return soundfile.read(wav_path, dtype="float32")


def load_16k_mono(path):
    """soundfile first; librosa for formats libsndfile can't read (e.g. m4a); ffmpeg as last resort."""
    try:
        wave, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
    except soundfile.LibsndfileError:
        try:
            import librosa
            wave, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            wave, sample_rate = decode_with_ffmpeg(path)
    return resample(wave, sample_rate, SAMPLE_RATE).astype(np.float32)


def rms(wave):
    return float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))


def avoid_clipping(wave):
    peak = np.max(np.abs(wave), initial=0.0)
    return wave / peak if peak > 1.0 else wave


def mix_at_snr(clean_wave, noise_wave, snr_db):
    """Scale noise so 10*log10(P_clean/P_noise) == snr_db, then add."""
    noise_scale = rms(clean_wave) / (rms(noise_wave) * 10 ** (snr_db / 20))
    return avoid_clipping(clean_wave + noise_scale * noise_wave)


def pick_babble_sources(babble_pool, item_id, target_speaker, rng):
    """One random utterance from each of 5 random distinct speakers, never the target utterance or speaker."""
    eligible_sources = [source for source in babble_pool
                        if source.utt_id != item_id and source.speaker_id != target_speaker]
    speaker_ids = sorted({source.speaker_id for source in eligible_sources})
    if len(speaker_ids) < BABBLE_TALKERS:
        raise ValueError(f"babble needs {BABBLE_TALKERS} eligible speakers, got {len(speaker_ids)}")
    picked_speakers = rng.choice(speaker_ids, size=BABBLE_TALKERS, replace=False)
    speaker_sources = [[source for source in eligible_sources if source.speaker_id == speaker_id]
                       for speaker_id in picked_speakers]
    return tuple(sources[rng.integers(len(sources))] for sources in speaker_sources)


def babble_noise(target_length, babble_sources, rng):
    """Sum of RMS-normalised sources, each looped/cropped to target_length from a random offset."""
    return sum(np.take(source.wave, np.arange(target_length) + rng.integers(len(source.wave)), mode="wrap")
               / rms(source.wave) for source in babble_sources)


def mu_law_encode(wave):
    """[-1, 1] floats to 8-bit codes 0..255 (mu=255)."""
    clipped_wave = np.clip(wave, -1.0, 1.0)
    compressed_wave = np.sign(clipped_wave) * np.log1p(MU * np.abs(clipped_wave)) / np.log1p(MU)
    return np.round((compressed_wave + 1) / 2 * MU).astype(np.uint8)


def mu_law_decode(mu_law_codes):
    compressed_wave = mu_law_codes.astype(np.float64) / MU * 2 - 1
    return np.sign(compressed_wave) * np.expm1(np.abs(compressed_wave) * np.log1p(MU)) / MU


def telephony(wave):
    """8 kHz, 300-3400 Hz band-pass, 8-bit mu-law, back to 16 kHz."""
    narrowband_wave = resample(wave, SAMPLE_RATE, TELEPHONY_RATE)
    band_pass = butter(6, (300, 3400), btype="bandpass", fs=TELEPHONY_RATE, output="sos")
    codec_wave = mu_law_decode(mu_law_encode(sosfiltfilt(band_pass, narrowband_wave)))
    return avoid_clipping(resample(codec_wave, TELEPHONY_RATE, SAMPLE_RATE)[: len(wave)])


def parse_condition(condition):
    steps = condition.split("+")
    bad_steps = [step for step in steps if not STEP_PATTERN.fullmatch(step)]
    if bad_steps:
        raise ValueError(f"unknown condition step(s) {bad_steps} in {condition!r}")
    return tuple(steps)


def apply_step(wave, step, item_id, babble_pool, target_speaker, rng):
    step_match = STEP_PATTERN.fullmatch(step)
    noise_kind, snr_text = step_match.group(1), step_match.group(2)
    if step == "clean":
        return wave
    if step == "telephony":
        return telephony(wave)
    if noise_kind == "white":
        return mix_at_snr(wave, rng.standard_normal(len(wave)), float(snr_text))
    babble_sources = pick_babble_sources(babble_pool, item_id, target_speaker, rng)
    return mix_at_snr(wave, babble_noise(len(wave), babble_sources, rng), float(snr_text))


def apply_condition(wave, condition, item_id, babble_pool, target_speaker=None):
    """Deterministic: each step's RNG derives from sha256(f"{item_id}|{condition}") and the step index."""
    steps = parse_condition(condition)
    base_seed = int.from_bytes(hashlib.sha256(f"{item_id}|{condition}".encode("utf-8")).digest()[:8], "little")
    degraded_wave = np.asarray(wave, dtype=np.float32)
    for step_index, step in enumerate(steps):
        step_rng = np.random.default_rng([base_seed, step_index])
        degraded_wave = apply_step(degraded_wave, step, item_id, babble_pool, target_speaker, step_rng)
    return degraded_wave.astype(np.float32)
