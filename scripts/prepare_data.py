"""Build the evaluation data: ASR track (Kathbath/FLEURS), QA questions (IndicQA/SQuAD), QA audio (TTS), examples.

Which languages, and which source each one uses, comes from languages.toml.

    PYTHONPATH=. uv run python scripts/prepare_data.py --stage asr|qa|examples
    PYTHONPATH=. uv run --project tts python scripts/prepare_data.py --stage tts

Heavy imports live inside the stage functions: the tts stage runs in the separate tts env.
"""

import argparse
import hashlib
import json
import random
import re
import tarfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from indic_s2t.languages import LANGUAGES

SAMPLE_RATE = 16000

# Kathbath's HF repo only carries train/valid; the test sets are AI4Bharat's tar downloads (IndicSUPERB README).
KATHBATH_URL = "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean"
KATHBATH_SPLITS = {  # split -> (audio tar, directory inside the tars)
    "test_unknown": ("testunk_audio.tar", "test"),
    "test_known": ("testkn_audio.tar", "test_known"),
    "valid": ("valid_audio.tar", "valid"),
}
FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
FLEURS_SPLIT = "test"
ASR_PER_LANG = 50
BABBLE_PER_LANG = 20
MIN_SECONDS, MAX_SECONDS = 1.0, 20.0

QA_ANSWERABLE, QA_UNANSWERABLE = 48, 12
MAX_CONTEXT_CHARS = 1500
MIN_WORDS, MAX_WORDS = 3, 20
QA_QUESTION_FIELDS = ("id", "track", "lang", "ref_text", "context", "answers", "answerable", "src_id")
MANIFEST_FIELDS = QA_QUESTION_FIELDS + ("audio", "duration", "tts_description")
QA_TRACK_FIELDS = MANIFEST_FIELDS + ("tts_attempts",)

TTS_MODEL = "ai4bharat/indic-parler-tts"
TTS_MAX_ATTEMPTS = 3
TTS_MIN_SECONDS = 0.8
TTS_MIN_CHARS_PER_SECOND, TTS_MAX_CHARS_PER_SECOND = 4.0, 30.0
ETA_WINDOW = 10
TTS_PACES = ("a normal", "a slightly fast")

EXAMPLE_CONDITIONS = ("clean", "babble_10db", "telephony")
EXAMPLES_PER_TRACK = 3


# ---------- shared io ----------

def read_jsonl(path):
    with open(path, encoding="utf-8") as jsonl_file:
        return tuple(json.loads(line) for line in jsonl_file if line.strip())


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as jsonl_file:
        jsonl_file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temp_path.replace(path)


def write_wav(path, wave):
    import soundfile
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, wave, SAMPLE_RATE, subtype="PCM_16")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "little")


# ---------- asr stage (Kathbath, FLEURS) ----------

@dataclass(frozen=True)
class Utterance:
    fname: str
    text: str
    speaker_id: str
    gender: str
    path: Path


def kathbath_speaker_id(fname):
    """Kathbath audio is named `{utterance}-{speaker}-{gender}.m4a`."""
    return Path(fname).stem.split("-")[1]


def asr_row(lang, index, audio_path, duration, ref_text, source, meta):
    return {"id": f"{lang}_{index:03d}", "track": "asr", "lang": lang, "audio": str(audio_path),
            "duration": round(duration, 3), "ref_text": ref_text, "source": source, "meta": meta}


def download(url, path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, open(part_path, "wb") as part_file:
        while chunk := response.read(1 << 20):
            part_file.write(chunk)
    part_path.replace(path)
    return path


def print_hf_kathbath_layout():
    from huggingface_hub import list_repo_files
    repo_files = list_repo_files("ai4bharat/Kathbath", repo_type="dataset")
    config_splits = Counter((path.split("/")[0], path.split("/")[1].split("-")[0])
                            for path in repo_files if path.endswith(".parquet"))
    print("HF ai4bharat/Kathbath configs/splits (parquet shards):")
    for (config, split), shard_count in sorted(config_splits.items()):
        print(f"  {config}/{split}: {shard_count}")
    print("  -> no test_known/test_unknown on HF; using AI4Bharat's official tar downloads for test splits")


def read_kathbath_transcripts(transcripts_tar, lang, split_dir, audio_root):
    member_name = f"kb_data_clean_m4a/{LANGUAGES[lang].kathbath}/{split_dir}/transcription_n2w.txt"
    with tarfile.open(transcripts_tar) as tar:
        transcript_text = tar.extractfile(member_name).read().decode("utf-8")
    fields = [line.split("\t", 1) for line in transcript_text.splitlines() if line.strip()]
    print(f"[{lang}] {member_name}: {len(fields)} rows, columns = (fname, text) tab-separated; "
          f"speaker/gender parsed from fname", flush=True)
    return tuple(Utterance(fname, text.strip(), kathbath_speaker_id(fname), Path(fname).stem.split("-")[2],
                           audio_root / LANGUAGES[lang].kathbath / fname)
                 for fname, text in fields)


def stream_tar_members(tar_url, member_paths, mode="r|"):
    """Stream a tar once over HTTP, writing only the wanted members (member name -> target path); no tar on disk."""
    missing_names = {member_name: path for member_name, path in member_paths.items() if not path.exists()}
    if not missing_names:
        return
    print(f"streaming {tar_url} for {len(missing_names)} files", flush=True)
    with urllib.request.urlopen(tar_url) as response, tarfile.open(fileobj=response, mode=mode) as tar:
        for member_index, member in enumerate(tar):
            target_path = missing_names.pop(member.name, None)
            if target_path is not None:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                part_path = target_path.with_suffix(".part")
                part_path.write_bytes(tar.extractfile(member).read())
                part_path.replace(target_path)
            if member_index % 2000 == 0:
                print(f"  {member_index} tar members read, {len(missing_names)} wanted files left", flush=True)
            if not missing_names:
                break
    if missing_names:
        raise RuntimeError(f"{len(missing_names)} files not found in {tar_url}, e.g. {next(iter(missing_names))}")


def kathbath_utterances(raw_dir, split):
    """lang -> utterances for every language with a `kathbath` directory; one pass over the shared audio tar."""
    kathbath_langs = tuple(lang for lang, language in LANGUAGES.items() if language.kathbath)
    if not kathbath_langs:
        return {}
    print_hf_kathbath_layout()
    audio_tar_name, split_dir = KATHBATH_SPLITS[split]
    transcripts_tar = download(f"{KATHBATH_URL}/transcripts_n2w.tar", raw_dir / "transcripts_n2w.tar")
    utterances_by_lang = {lang: read_kathbath_transcripts(transcripts_tar, lang, split_dir, raw_dir / split)
                          for lang in kathbath_langs}
    stream_tar_members(f"{KATHBATH_URL}/{audio_tar_name}",
                       {f"kb_data_clean_m4a/{utterance.path.parent.name}/{split_dir}/audio/{utterance.path.name}":
                        utterance.path for utterances in utterances_by_lang.values() for utterance in utterances})
    return utterances_by_lang


def parse_fleurs_tsv(tsv_text, audio_dir):
    """FLEURS `{split}.tsv` has no header: sentence id, file, raw transcription, transcription, characters, samples,
    gender. The raw (cased, punctuated) transcription is the reference; normalize() strips both sides alike."""
    # ponytail: FLEURS has no speaker ids; each recording counts as its own speaker, so babble may share a voice
    # with the target.
    fields = [line.split("\t") for line in tsv_text.splitlines() if line.strip()]
    return tuple(Utterance(fname, raw_text.strip(), Path(fname).stem, gender.lower(), audio_dir / fname)
                 for _, fname, raw_text, *_, gender in fields)


def read_fleurs(config, raw_dir):
    tsv_path = download(f"{FLEURS_URL}/{config}/{FLEURS_SPLIT}.tsv", raw_dir / config / f"{FLEURS_SPLIT}.tsv")
    utterances = parse_fleurs_tsv(tsv_path.read_text(encoding="utf-8"), raw_dir / config / FLEURS_SPLIT)
    print(f"[{config}] {tsv_path}: {len(utterances)} rows", flush=True)
    stream_tar_members(f"{FLEURS_URL}/{config}/audio/{FLEURS_SPLIT}.tar.gz",
                       {f"{FLEURS_SPLIT}/{utterance.path.name}": utterance.path for utterance in utterances},
                       mode="r|gz")
    return utterances


def asr_source(lang, split):
    return f"kathbath/{split}" if LANGUAGES[lang].kathbath else f"fleurs/{LANGUAGES[lang].fleurs}/{FLEURS_SPLIT}"


def assign_roles(decoded_utterances, asr_count, babble_count):
    """decoded_utterances: (utterance, wave) pairs in shuffled order, consumed lazily.
    The first asr_count in [1, 20] s go to ASR; after that, one utterance per new speaker goes to babble."""
    asr_picks, babble_picks = [], []
    for utterance, wave in decoded_utterances:
        if not MIN_SECONDS <= len(wave) / SAMPLE_RATE <= MAX_SECONDS:
            continue
        if len(asr_picks) < asr_count:
            asr_picks.append((utterance, wave))
        elif utterance.speaker_id not in {picked.speaker_id for picked, _ in babble_picks}:
            babble_picks.append((utterance, wave))
        if len(asr_picks) == asr_count and len(babble_picks) == babble_count:
            break
    if len(asr_picks) < asr_count:
        raise RuntimeError(f"only {len(asr_picks)} utterances within {MIN_SECONDS}-{MAX_SECONDS} s")
    return tuple(asr_picks), tuple(babble_picks)


def run_asr(out_dir, seed, split):
    from indic_s2t.audio import load_16k_mono

    utterances_by_lang = {**kathbath_utterances(out_dir / "raw" / "kathbath", split),
                          **{lang: read_fleurs(language.fleurs, out_dir / "raw" / "fleurs")
                             for lang, language in LANGUAGES.items() if language.fleurs}}

    asr_rows, babble_rows = [], []
    for lang in LANGUAGES:
        if lang not in utterances_by_lang:
            print(f"[{lang}] no ASR source (kathbath/fleurs), skipped")
            continue
        utterances = utterances_by_lang[lang]
        print(f"[{lang}] {len(utterances)} utterances from {len({u.speaker_id for u in utterances})} speakers")
        shuffled_utterances = random.Random(f"{seed}:{lang}").sample(sorted(utterances, key=lambda u: u.fname),
                                                                     len(utterances))
        asr_picks, babble_picks = assign_roles(((utterance, load_16k_mono(utterance.path))
                                                for utterance in shuffled_utterances), ASR_PER_LANG, BABBLE_PER_LANG)
        for index, (utterance, wave) in enumerate(asr_picks):
            audio_path = out_dir / "asr" / f"{lang}_{index:03d}.wav"
            write_wav(audio_path, wave)
            asr_rows.append(asr_row(lang, index, audio_path, len(wave) / SAMPLE_RATE, utterance.text,
                                    asr_source(lang, split),
                                    {"speaker_id": utterance.speaker_id, "gender": utterance.gender,
                                     "original_id": Path(utterance.fname).stem}))
        for index, (utterance, wave) in enumerate(babble_picks):
            audio_path = out_dir / "babble" / f"{lang}_{index:03d}.wav"
            write_wav(audio_path, wave)
            babble_rows.append({"utt_id": Path(utterance.fname).stem, "lang": lang,
                                "speaker_id": utterance.speaker_id, "audio": str(audio_path)})
        durations = sorted(len(wave) / SAMPLE_RATE for _, wave in asr_picks)
        print(f"[{lang}] asr: {len(asr_picks)} utterances, {len({u.speaker_id for u, _ in asr_picks})} speakers, "
              f"duration min/median/max = {durations[0]:.1f}/{durations[len(durations) // 2]:.1f}/"
              f"{durations[-1]:.1f} s, total {sum(durations) / 60:.1f} min; babble: {len(babble_picks)} "
              f"utterances from distinct speakers", flush=True)
    write_jsonl(out_dir / "asr_track.jsonl", asr_rows)
    write_jsonl(out_dir / "babble_pool.jsonl", babble_rows)
    print(f"wrote {len(asr_rows)} asr rows, {len(babble_rows)} babble rows")


# ---------- qa stage (IndicQA, SQuAD) ----------

@dataclass(frozen=True)
class QAQuestion:
    src_id: str
    passage_id: object
    question: str
    context: str
    answers: tuple
    answerable: bool


def parse_squad(squad_json):
    """-> (questions, malformed_count). IndicQA marks unanswerable questions with a single empty-text answer."""
    questions, malformed_count = [], 0
    for article_index, article in enumerate(squad_json.get("data", [])):
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            for qa in paragraph.get("qas", []):
                try:
                    answer_texts = tuple(answer["text"].strip() for answer in qa["answers"] if answer["text"].strip())
                    questions.append(QAQuestion(
                        src_id=f"{article_index}.{paragraph_index}.{qa['id']}",
                        passage_id=(article_index, paragraph_index), question=qa["question"].strip(),
                        context=paragraph["context"], answers=answer_texts, answerable=bool(answer_texts)))
                except (KeyError, TypeError, AttributeError):
                    malformed_count += 1
    return tuple(questions), malformed_count


def word_count(text):
    return sum(any(char.isalnum() for char in token) for token in text.split())


def has_no_digits_and_only_own_script(question, lang):
    from indic_s2t.textnorm import script_letter_share  # textnorm needs indic-nlp, which the tts env lacks
    return not re.search(r"\d", question.question) and script_letter_share(question.question, lang) == 1.0


# Ordered: counts are for what each filter removes after the previous ones. `\d` matches Indic digits too.
QUESTION_FILTERS = (
    ("digits_or_off_script", has_no_digits_and_only_own_script),
    ("word_count", lambda question, lang: MIN_WORDS <= word_count(question.question) <= MAX_WORDS),
    ("passage_length", lambda question, lang: len(question.context) <= MAX_CONTEXT_CHARS),
)


def apply_question_filters(questions, lang):
    removed_counts = []
    for filter_name, keep in QUESTION_FILTERS:
        kept_questions = tuple(question for question in questions if keep(question, lang))
        removed_counts.append((filter_name, len(questions) - len(kept_questions)))
        questions = kept_questions
    return questions, tuple(removed_counts)


def sample_round_robin(questions, answerable_count, unanswerable_count, seed):
    """Seeded round-robin over passages: every passage gives one question per round, within per-class quotas."""
    rng = random.Random(seed)
    passage_ids = sorted({question.passage_id for question in questions})
    passage_groups = [[q for q in questions if q.passage_id == passage_id] for passage_id in passage_ids]
    passage_queues = [rng.sample(group, len(group)) for group in passage_groups]
    rng.shuffle(passage_queues)
    round_robin_order = sorted(((rank, queue_index, question) for queue_index, queue in enumerate(passage_queues)
                                for rank, question in enumerate(queue)), key=lambda entry: entry[:2])
    quotas = {True: answerable_count, False: unanswerable_count}
    picked_questions = []
    for _, _, question in round_robin_order:
        if sum(picked.answerable == question.answerable for picked in picked_questions) < quotas[question.answerable]:
            picked_questions.append(question)
    if len(picked_questions) < answerable_count + unanswerable_count:
        raise ValueError(f"pool too small: wanted {answerable_count}+{unanswerable_count}, "
                         f"got {len(picked_questions)}")
    return tuple(picked_questions)


def question_rows(questions, lang):
    return tuple({"id": f"qa_{lang}_{index:03d}", "track": "qa", "lang": lang, "ref_text": question.question,
                  "context": question.context, "answers": list(question.answers),
                  "answerable": question.answerable, "src_id": question.src_id}
                 for index, question in enumerate(questions))


def squad_file_for(lang, raw_dir):
    """Local SQuAD-format JSON for the language's QA source, or None when it has none."""
    language = LANGUAGES[lang]
    if language.indicqa:
        from huggingface_hub import hf_hub_download
        return hf_hub_download("ai4bharat/IndicQA", f"data/indicqa.{language.indicqa}.json", repo_type="dataset")
    if language.squad_url:
        return download(language.squad_url, raw_dir / "squad" / f"{lang}.json")
    return None


def run_qa(out_dir, seed):
    all_rows = []
    for lang in LANGUAGES:
        squad_path = squad_file_for(lang, out_dir / "raw")
        if squad_path is None:
            print(f"[{lang}] no QA source (indicqa/squad_url), skipped")
            continue
        with open(squad_path, encoding="utf-8") as squad_file:
            questions, malformed_count = parse_squad(json.load(squad_file))
        kept_questions, removed_counts = apply_question_filters(questions, lang)
        print(f"[{lang}] {len(questions)} questions parsed, {malformed_count} malformed skipped; unanswerable: "
              f"{sum(not q.answerable for q in questions)}")
        for filter_name, removed_count in removed_counts:
            print(f"[{lang}]   filter {filter_name}: removed {removed_count}")
        print(f"[{lang}] kept {len(kept_questions)} ({sum(q.answerable for q in kept_questions)} answerable, "
              f"{sum(not q.answerable for q in kept_questions)} unanswerable) from "
              f"{len({q.passage_id for q in kept_questions})} passages")
        sampled_questions = sample_round_robin(kept_questions, QA_ANSWERABLE, QA_UNANSWERABLE, f"{seed}:{lang}")
        print(f"[{lang}] sampled {len(sampled_questions)}: {sum(q.answerable for q in sampled_questions)} answerable, "
              f"{sum(not q.answerable for q in sampled_questions)} unanswerable, "
              f"{len({q.passage_id for q in sampled_questions})} distinct passages", flush=True)
        all_rows.extend(question_rows(sampled_questions, lang))
    write_jsonl(out_dir / "qa_questions.jsonl", all_rows)
    print(f"wrote {len(all_rows)} rows to {out_dir / 'qa_questions.jsonl'}")


# ---------- tts stage (Parler-TTS) ----------

def passes_tts_qc(question_text, duration):
    return duration >= TTS_MIN_SECONDS and \
        TTS_MIN_CHARS_PER_SECOND <= len(question_text) / duration <= TTS_MAX_CHARS_PER_SECOND


def voice_description(lang, index):
    """Rotates female/male x normal/slightly-fast: index 0 F normal, 1 M normal, 2 F fast, 3 M fast."""
    speaker = LANGUAGES[lang].tts_speakers[index % 2]
    pace = TTS_PACES[(index // 2) % 2]
    return (f"{speaker} speaks at {pace} pace with a moderate pitch. "
            f"The recording is of very clear audio, close-sounding, with no background noise.")


def eta_seconds(recent_item_seconds, remaining_count):
    window = recent_item_seconds[-ETA_WINDOW:]
    return sum(window) / len(window) * remaining_count if window else None


def qa_track_row(question_row, audio_path, duration, tts_description, tts_attempts):
    return {**{field: question_row[field] for field in QA_QUESTION_FIELDS}, "audio": str(audio_path),
            "duration": round(duration, 3), "tts_description": tts_description, "tts_attempts": tts_attempts}


def validate_manifest_row(row):
    missing_fields = [field for field in MANIFEST_FIELDS if field not in row]
    if missing_fields:
        raise ValueError(f"manifest row {row.get('id')} is missing fields {missing_fields}")


def import_manifest(manifest_path, out_dir):
    """Hand-recorded QA audio: validate rows, normalise audio to data/qa/{id}.wav at 16 kHz mono."""
    from indic_s2t.audio import load_16k_mono

    track_rows = []
    for row in read_jsonl(manifest_path):
        validate_manifest_row(row)
        source_path = next((path for path in (Path(row["audio"]), manifest_path.parent / row["audio"])
                            if path.exists()), None)
        if source_path is None:
            raise FileNotFoundError(f"{row['id']}: audio {row['audio']} not found")
        wave = load_16k_mono(source_path)
        audio_path = out_dir / "qa" / f"{row['id']}.wav"
        write_wav(audio_path, wave)
        track_rows.append(qa_track_row(row, audio_path, len(wave) / SAMPLE_RATE, row["tts_description"],
                                       row.get("tts_attempts")))
    write_jsonl(out_dir / "qa_track.jsonl", track_rows)
    print(f"imported {len(track_rows)} rows from {manifest_path}")


def run_tts(out_dir, seed):
    import numpy as np
    import soundfile
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    from indic_s2t.audio import resample

    torch.set_num_threads(8)
    question_rows_in = read_jsonl(out_dir / "qa_questions.jsonl")
    track_path = out_dir / "qa_track.jsonl"
    previous_rows = {row["id"]: row for row in read_jsonl(track_path)} if track_path.exists() else {}
    model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=torch.float32).eval()
    prompt_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    def synthesize(text, description, generation_seed):
        torch.manual_seed(generation_seed)
        description_inputs = description_tokenizer(description, return_tensors="pt")
        prompt_inputs = prompt_tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            generation = model.generate(input_ids=description_inputs.input_ids,
                                        attention_mask=description_inputs.attention_mask,
                                        prompt_input_ids=prompt_inputs.input_ids,
                                        prompt_attention_mask=prompt_inputs.attention_mask)
        return resample(generation.cpu().numpy().squeeze().astype(np.float32),
                        model.config.sampling_rate, SAMPLE_RATE).astype(np.float32)

    track_rows, recent_item_seconds = [], ()
    for position, row in enumerate(question_rows_in):
        audio_path = out_dir / "qa" / f"{row['id']}.wav"
        description = voice_description(row["lang"], int(row["id"].rsplit("_", 1)[1]))
        if audio_path.exists():
            duration = soundfile.info(audio_path).duration
            attempts = previous_rows.get(row["id"], {}).get("tts_attempts")
            track_rows.append(qa_track_row(row, audio_path, duration, description, attempts))
            continue
        started = time.monotonic()
        for attempts in range(1, TTS_MAX_ATTEMPTS + 1):
            wave = synthesize(row["ref_text"], description, stable_seed(seed, row["id"], attempts))
            duration = len(wave) / SAMPLE_RATE
            if passes_tts_qc(row["ref_text"], duration):
                break
            print(f"  {row['id']} attempt {attempts} failed QC ({duration:.2f} s, "
                  f"{len(row['ref_text']) / max(duration, 1e-6):.1f} chars/s)", flush=True)
        write_wav(audio_path, wave)  # after 3 failed attempts the last one is kept; tts_attempts records it
        track_rows.append(qa_track_row(row, audio_path, duration, description, attempts))
        write_jsonl(track_path, track_rows)  # rewritten each item so a crash keeps the attempts recorded so far
        recent_item_seconds += (time.monotonic() - started,)
        remaining_eta = eta_seconds(recent_item_seconds, len(question_rows_in) - position - 1)
        print(f"[{position + 1}/{len(question_rows_in)}] {row['id']} {duration:.1f} s, attempts {attempts}, "
              f"{recent_item_seconds[-1]:.1f} s/item, ETA {remaining_eta / 60:.1f} min", flush=True)
    write_jsonl(track_path, track_rows)
    print(f"wrote {len(track_rows)} rows to {track_path}")


# ---------- examples stage ----------

def run_examples(out_dir):
    from indic_s2t.audio import BabbleSource, apply_condition, load_16k_mono

    asr_track_path = out_dir / "asr_track.jsonl"
    if not asr_track_path.exists():
        print(f"{asr_track_path} missing: run --stage asr first")
        return
    babble_pool = tuple(BabbleSource(row["utt_id"], row["speaker_id"], load_16k_mono(row["audio"]))
                        for row in read_jsonl(out_dir / "babble_pool.jsonl"))
    babble_lang = {row["utt_id"]: row["lang"] for row in read_jsonl(out_dir / "babble_pool.jsonl")}
    qa_path = next((path for path in (out_dir / "qa_track.jsonl", out_dir / "qa_questions.jsonl") if path.exists()),
                   None)
    track_rows = read_jsonl(asr_track_path) + (read_jsonl(qa_path) if qa_path else ())
    for lang in LANGUAGES:
        lang_pool = tuple(source for source in babble_pool if babble_lang[source.utt_id] == lang)
        for track in ("asr", "qa"):
            for row in [row for row in track_rows if row["lang"] == lang and row["track"] == track][:EXAMPLES_PER_TRACK]:
                if not Path(row.get("audio", "")).is_file():
                    print(f"skip {row['id']}: no audio yet")
                    continue
                wave = load_16k_mono(row["audio"])
                target_speaker = row.get("meta", {}).get("speaker_id")
                for condition in EXAMPLE_CONDITIONS:
                    write_wav(out_dir / "examples" / condition / f"{row['id']}.wav",
                              apply_condition(wave, condition, row["id"], lang_pool, target_speaker))
                print(f"wrote examples for {row['id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("asr", "qa", "tts", "examples"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--kathbath_split", default="test_unknown", choices=tuple(KATHBATH_SPLITS))
    parser.add_argument("--manifest", type=Path, help="tts stage: hand-recorded QA manifest instead of TTS")
    args = parser.parse_args()
    if args.stage == "asr":
        run_asr(args.out, args.seed, args.kathbath_split)
    elif args.stage == "qa":
        run_qa(args.out, args.seed)
    elif args.stage == "tts" and args.manifest:
        import_manifest(args.manifest, args.out)
    elif args.stage == "tts":
        run_tts(args.out, args.seed)
    else:
        run_examples(args.out)


if __name__ == "__main__":
    main()
