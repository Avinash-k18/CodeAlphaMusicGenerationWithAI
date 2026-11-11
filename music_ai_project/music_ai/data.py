"""Step 1 + 2: collect MIDI files and preprocess them into note sequences."""
import json, random
from collections import Counter
from pathlib import Path

import numpy as np
from music21 import chord, converter, corpus, note

ART = Path("artifacts")
MIDI_EXT = {".mid", ".midi"}


def quantize(ql):
    """Round a duration (in quarter notes) to a 0.25 grid, clamped to [0.25, 4]."""
    try:
        q = round(float(ql) * 4) / 4
    except Exception:
        return 0.5
    return min(max(q, 0.25), 4.0)


def collect_files(midi_dir, max_files, seed=42):
    """Use your own MIDI files if present; otherwise fall back to music21's built-in Bach corpus."""
    d = Path(midi_dir)
    files = [str(p) for p in d.rglob("*") if p.suffix.lower() in MIDI_EXT] if d.is_dir() else []
    if files:
        print(f"Found {len(files)} MIDI files in {d}")
    else:
        print(f"No MIDI files in '{d}' -> using music21's built-in Bach corpus (no download needed).")
        files = [str(p) for p in corpus.getComposer("bach")]
    random.Random(seed).shuffle(files)
    return files[:max_files]


def file_to_tokens(path):
    """Parse one file into tokens like '60_0.5' (note) or '60.64.67_1.0' (chord). Returns [] on any failure."""
    try:
        score = converter.parse(path)
        parts = list(score.parts) if score.parts else [score]
        best = max(parts, key=lambda p: len(p.flatten().notes))  # the busiest track = melody
        tokens = []
        for el in best.flatten().notes:
            if isinstance(el, note.Note):
                pitch = str(el.pitch.midi)
            elif isinstance(el, chord.Chord):
                pitch = ".".join(str(p.midi) for p in el.pitches)
            else:
                continue
            tokens.append(f"{pitch}_{quantize(el.duration.quarterLength)}")
        return tokens
    except Exception as e:  # corrupt / unsupported file: skip, never crash
        print(f"  skipped {Path(path).name}: {type(e).__name__}")
        return []


def build_dataset(midi_dir="data/midi", max_files=60, seq_len=32, min_count=3):
    ART.mkdir(exist_ok=True)
    files = collect_files(midi_dir, max_files)
    seqs = []
    for i, f in enumerate(files, 1):
        toks = file_to_tokens(f)
        print(f"[{i}/{len(files)}] {Path(f).name}: {len(toks)} notes")
        if len(toks) > seq_len + 1:  # too-short pieces cannot form a single training window
            seqs.append(toks)
    if not seqs:
        raise SystemExit("No usable MIDI data found. Add .mid files to data/midi/ and retry.")

    counts = Counter(t for s in seqs for t in s)
    vocab = sorted(t for t, c in counts.items() if c >= min_count)  # drop very rare tokens
    if len(vocab) < 5:
        raise SystemExit("Vocabulary too small - add more/different MIDI files.")
    idx = {t: i for i, t in enumerate(vocab)}

    X, y, seeds = [], [], []
    for s in seqs:
        ids = [idx[t] for t in s if t in idx]
        if len(ids) <= seq_len:
            continue
        seeds.append(ids[:seq_len])
        for i in range(len(ids) - seq_len):
            X.append(ids[i:i + seq_len])
            y.append(ids[i + seq_len])
    if len(X) < 200:
        raise SystemExit(f"Only {len(X)} training windows - too little data. Add more MIDI files.")

    X, y = np.array(X, dtype="int32"), np.array(y, dtype="int32")
    random.Random(1).shuffle(seeds)
    meta = {"vocab": vocab, "seq_len": seq_len, "seeds": seeds[:300]}
    (ART / "meta.json").write_text(json.dumps(meta))
    np.savez_compressed(ART / "dataset.npz", X=X, y=y)
    print(f"Dataset: {len(X)} windows | vocab {len(vocab)} | {len(seqs)} pieces")
    return X, y, meta
