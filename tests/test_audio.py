import numpy as np
import pytest
import soundfile

from indic_s2t import audio

SAMPLE_RATE = 16000


def tone(frequency, seconds=2.0, amplitude=0.1):
    sample_times = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * sample_times)).astype(np.float32)


def make_babble_pool(speaker_count=6, amplitude=0.1):
    """One utterance per speaker plus a second utterance for speaker s0, each a distinct tone."""
    sources = tuple(audio.BabbleSource(f"utt{index}", f"s{index}", tone(200 + 50 * index, 1.3, amplitude))
                    for index in range(speaker_count))
    return sources + (audio.BabbleSource("utt_extra", "s0", tone(170, 0.7, amplitude)),)


def snr_db(clean_wave, noise_wave):
    return 10 * np.log10(np.sum(clean_wave.astype(np.float64) ** 2) / np.sum(noise_wave.astype(np.float64) ** 2))


def test_same_args_give_identical_output():
    target_wave = tone(440)
    babble_pool = make_babble_pool()
    first_output = audio.apply_condition(target_wave, "babble_10db+telephony", "utt_target", babble_pool, "s_target")
    second_output = audio.apply_condition(target_wave, "babble_10db+telephony", "utt_target", babble_pool, "s_target")
    assert np.array_equal(first_output, second_output)


def test_different_condition_or_item_id_changes_output():
    target_wave = tone(440)
    babble_pool = make_babble_pool()
    reference_output = audio.apply_condition(target_wave, "white_10db", "utt_a", babble_pool)
    assert not np.array_equal(reference_output, audio.apply_condition(target_wave, "white_10db", "utt_b", babble_pool))
    assert not np.array_equal(reference_output, audio.apply_condition(target_wave, "white_5db", "utt_a", babble_pool))


def test_clean_is_identity():
    target_wave = tone(440)
    assert np.array_equal(audio.apply_condition(target_wave, "clean", "utt_a", ()), target_wave)


@pytest.mark.parametrize("condition, target_snr", [("babble_10db", 10.0), ("babble_0db", 0.0), ("white_20db", 20.0)])
def test_mixture_snr_matches_request(condition, target_snr):
    target_wave = tone(440, amplitude=0.05)
    mixed_wave = audio.apply_condition(target_wave, condition, "utt_target", make_babble_pool(amplitude=0.05))
    assert abs(snr_db(target_wave, mixed_wave - target_wave) - target_snr) < 0.5


def test_loud_mixture_is_scaled_down_but_keeps_snr():
    target_wave = tone(440, amplitude=0.9)
    mixed_wave = audio.apply_condition(target_wave, "white_0db", "utt_target", ())
    assert np.max(np.abs(mixed_wave)) <= 1.0
    scale = np.dot(mixed_wave, target_wave) / np.dot(target_wave, target_wave)  # noise is ~orthogonal to the tone
    assert abs(snr_db(scale * target_wave, mixed_wave - scale * target_wave)) < 0.5


def test_pick_babble_sources_excludes_target_and_uses_distinct_speakers():
    babble_pool = make_babble_pool(speaker_count=7)
    for seed in range(20):
        picked_sources = audio.pick_babble_sources(babble_pool, "utt1", "s2", np.random.default_rng(seed))
        picked_speakers = [source.speaker_id for source in picked_sources]
        assert len(picked_sources) == 5
        assert len(set(picked_speakers)) == 5
        assert "s2" not in picked_speakers
        assert "utt1" not in [source.utt_id for source in picked_sources]


def test_babble_never_uses_excluded_huge_source():
    target_wave = tone(440, amplitude=0.05)
    huge_wave = np.full(SAMPLE_RATE, 50.0, dtype=np.float32)
    excluded_sources = (audio.BabbleSource("utt_target", "s_other", huge_wave),
                        audio.BabbleSource("utt_same_speaker", "s_target", huge_wave))
    babble_pool = make_babble_pool(speaker_count=5) + excluded_sources
    mixed_wave = audio.apply_condition(target_wave, "babble_10db", "utt_target", babble_pool, "s_target")
    babble_wave = mixed_wave - target_wave
    # a DC source would dominate the mean if it were mixed in; the tones average to ~0
    assert abs(np.mean(babble_wave)) < 1e-3


def test_babble_needs_five_eligible_speakers():
    babble_pool = make_babble_pool(speaker_count=5)  # s0..s4; excluding s0 leaves four speakers
    with pytest.raises(ValueError):
        audio.apply_condition(tone(440), "babble_10db", "utt_target", babble_pool, "s0")


def test_telephony_removes_energy_above_3400_hz():
    broadband_wave = (0.1 * np.random.default_rng(0).standard_normal(2 * SAMPLE_RATE)).astype(np.float32)
    phone_wave = audio.apply_condition(broadband_wave, "telephony", "utt_a", ())
    assert len(phone_wave) == len(broadband_wave)
    power_spectrum = np.abs(np.fft.rfft(phone_wave)) ** 2
    frequencies = np.fft.rfftfreq(len(phone_wave), 1 / SAMPLE_RATE)
    assert power_spectrum[frequencies > 3400].sum() / power_spectrum.sum() < 0.01


def test_mu_law_round_trip_error_is_bounded():
    ramp_wave = np.linspace(-1, 1, 10001)
    mu_law_codes = audio.mu_law_encode(ramp_wave)
    assert mu_law_codes.min() >= 0 and mu_law_codes.max() <= 255
    decoded_wave = audio.mu_law_decode(mu_law_codes)
    # 8-bit mu-law: step is finest near zero and ~0.03 at full scale
    assert np.max(np.abs(decoded_wave - ramp_wave)) < 0.04
    quiet_wave = ramp_wave[np.abs(ramp_wave) < 0.01]
    assert np.max(np.abs(audio.mu_law_decode(audio.mu_law_encode(quiet_wave)) - quiet_wave)) < 0.001


def test_chained_condition_runs():
    mixed_wave = audio.apply_condition(tone(440), "telephony+white_20db", "utt_a", ())
    assert mixed_wave.dtype == np.float32 and len(mixed_wave) == 2 * SAMPLE_RATE


@pytest.mark.parametrize("bad_condition", ["reverb", "babble_db", "telephony+pink_10db", ""])
def test_unknown_condition_raises(bad_condition):
    with pytest.raises(ValueError):
        audio.apply_condition(tone(440), bad_condition, "utt_a", ())


def test_load_16k_mono_resamples_and_downmixes(tmp_path):
    stereo_wave = np.stack([tone(440, 1.0), tone(440, 1.0)], axis=1)
    resampled_source = np.repeat(stereo_wave, 3, axis=0)[: 48000]  # crude 48 kHz stand-in, 1 s long
    wav_path = tmp_path / "stereo48k.wav"
    soundfile.write(wav_path, resampled_source, 48000)
    loaded_wave = audio.load_16k_mono(wav_path)
    assert loaded_wave.dtype == np.float32 and loaded_wave.ndim == 1
    assert len(loaded_wave) == SAMPLE_RATE


def test_load_16k_mono_falls_back_for_m4a(tmp_path):
    wav_path = tmp_path / "tone.wav"
    m4a_path = tmp_path / "tone.m4a"
    soundfile.write(wav_path, tone(440, 1.0), SAMPLE_RATE)
    import subprocess
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(wav_path), str(m4a_path)], check=True)
    loaded_wave = audio.load_16k_mono(m4a_path)
    assert loaded_wave.dtype == np.float32
    assert abs(len(loaded_wave) - SAMPLE_RATE) < 0.05 * SAMPLE_RATE
