"""Fine-tuning data preparation for Accent Vector on F5-TTS.

Input is an ``audio_file|text`` metadata CSV, built per corpus by the scripts in
Datasets/prep-scripts/ (which also apply the paper's >= 3 s floor, Section 4.2).

``prepare`` -- tokenize transcripts and write the F5 Arrow dataset
   (``train.arrow``/``train_duration.json`` + ``valid.arrow``/``valid_duration.json``
   + ``vocab.txt``) that ``finetune_cli.py`` consumes -- this fork's trainer loads
   a pre-split train/valid pair (``load_dataset(..., audio_type="train"|"valid")``),
   not a single ``raw.arrow``. The split is controlled by ``--val-frac`` (default
   0.1, floored at 1 clip). Reuses the pretrained vocab for fine-tuning.

The tokenizer is F5's default ``pinyin`` map: Latin letters pass through
unchanged and Hanzi become pinyin. Other non-Latin scripts (Devanagari, Bengali,
Arabic) are NOT covered by the base vocab -- romanise the transcripts first
(Datasets/prep-scripts/indicvoices/romanize.py) and check with common/vocab_check.py.

Usage
-----
    python -m accent_vector.data_preprocess prepare \
        --metadata   /path/to/cgn_dutch_clips/metadata.dnsmos.csv \
        --audio-root /path/to/cgn_dutch_clips \
        --out-dir    $F5_ROOT/data/dutch_pinyin
"""

import argparse
import csv
import json
import multiprocessing
import os
import shutil
from importlib.resources import files
from pathlib import Path

import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin

# Env override lets you point at the base vocab without staging a file into the
# F5-TTS tree (e.g. F5_VOCAB=$F5_ROOT/examples/vocab.txt); default is unchanged.
PRETRAINED_VOCAB_PATH = os.environ.get("F5_VOCAB") or files("f5_tts").joinpath("../../data/vocab.txt")
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)


def audio_duration(path):
    """Clip duration in seconds. Prefers torchaudio.info (present on the cluster's
    pinned torch 2.4); falls back to soundfile on newer torchaudio (>=2.9 removed
    torchaudio.info), so data prep also runs on a local Mac env."""
    if hasattr(torchaudio, "info"):
        info = torchaudio.info(str(path))
        return info.num_frames / info.sample_rate
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


# --- metadata -> F5 Arrow dataset ----------------------------------
def read_audio_text_pairs(csv_path, audio_root):
    pairs = []
    audio_root = Path(audio_root)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader)  # header
        for row in reader:
            if len(row) >= 2:
                pairs.append(((audio_root / row[0].strip()).as_posix(), row[1].strip()))
    return pairs


def prepare_dataset(metadata_csv, audio_root, out_dir, is_finetune=True, lora_label=None,
                     val_frac=0.1):
    pairs = read_audio_text_pairs(metadata_csv, audio_root)
    if not pairs:
        raise RuntimeError(f"No usable rows in {metadata_csv}")

    result, durations, vocab_set = [], [], set()
    texts = [t for _, t in pairs]
    converted = []
    for i in range(0, len(texts), 100):  # tokenize in batches
        converted.extend(convert_char_to_pinyin(texts[i : i + 100], polyphone=True))

    for (audio_path, _), conv_text in tqdm(
        list(zip(pairs, converted)), desc="collecting"
    ):
        if not Path(audio_path).exists():
            print(f"[prepare] missing audio, skipping: {audio_path}")
            continue
        try:
            duration = audio_duration(audio_path)
        except Exception as e:
            print(f"[prepare] cannot read {audio_path}: {e}")
            continue
        if duration <= 0:
            continue
        row = {"audio_path": audio_path, "text": conv_text, "duration": duration}
        if lora_label is not None:
            # per-sample LoRA label; a single-accent run uses one constant label.
            # The LoRA dataset/sampler (dataset.py) reads row["lora_label"].
            row["lora_label"] = int(lora_label)
        result.append(row)
        durations.append(duration)
        vocab_set.update(list(conv_text))

    # split train/valid: this fork's finetune_cli.py loads pre-split
    # train.arrow/valid.arrow (+ matching *_duration.json), not a single raw.arrow.
    n_valid = max(1, round(len(result) * val_frac)) if len(result) > 1 else 0
    valid_result, valid_durations = result[:n_valid], durations[:n_valid]
    train_result, train_durations = result[n_valid:], durations[n_valid:]
    if not train_result:
        train_result, train_durations = result, durations
        valid_result, valid_durations = result, durations

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows, split_durations in (
        ("train", train_result, train_durations),
        ("valid", valid_result, valid_durations),
    ):
        with ArrowWriter(path=(out_dir / f"{split_name}.arrow").as_posix(), writer_batch_size=100) as writer:
            for line in tqdm(split_rows, desc=f"writing {split_name}.arrow"):
                writer.write(line)
            writer.finalize()  # write schema/footer; the `with` exit alone leaves a 0-byte file
        with open(out_dir / f"{split_name}_duration.json", "w", encoding="utf-8") as f:
            json.dump({"duration": split_durations}, f, ensure_ascii=False)

    vocab_out = out_dir / "vocab.txt"
    if is_finetune:
        assert Path(PRETRAINED_VOCAB_PATH).exists(), (
            f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
        )
        shutil.copy2(PRETRAINED_VOCAB_PATH, vocab_out)
    else:
        with open(vocab_out, "w", encoding="utf-8") as f:
            for v in sorted(vocab_set):
                f.write(v + "\n")

    print(
        f"[prepare] {out_dir.name}: {len(result)} samples "
        f"({len(train_result)} train / {len(valid_result)} valid), "
        f"{sum(durations) / 3600:.2f} h, vocab {len(vocab_set)} -> {out_dir}"
    )
    return out_dir


def _build_parser():
    parser = argparse.ArgumentParser(description="Accent Vector data preparation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_p = sub.add_parser("prepare", help="metadata -> F5 Arrow dataset")
    p_p.add_argument("--metadata", required=True, help="audio_file|text CSV")
    p_p.add_argument("--audio-root", required=True, help="prefix joined to CSV audio paths")
    p_p.add_argument("--out-dir", required=True)
    p_p.add_argument("--no-finetune-vocab", action="store_true",
                     help="build a fresh vocab instead of copying the pretrained one")
    p_p.add_argument("--lora-label", type=int, default=None,
                     help="write a constant per-sample lora_label (required for LoRA "
                          "training; a single-accent run uses one label, e.g. 0)")
    p_p.add_argument("--val-frac", type=float, default=0.1,
                     help="fraction of clips held out as train.arrow/valid.arrow split "
                          "(finetune_cli.py requires both; min 1 clip)")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "prepare":
        prepare_dataset(
            args.metadata, args.audio_root, args.out_dir,
            is_finetune=not args.no_finetune_vocab,
            lora_label=args.lora_label,
            val_frac=args.val_frac,
        )


if __name__ == "__main__":
    main()
