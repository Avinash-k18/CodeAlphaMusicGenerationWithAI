"""Step 5: sample new note sequences and convert them to MIDI."""
import json
from pathlib import Path

import numpy as np
from music21 import chord, instrument, note, stream, tempo

ART = Path("artifacts")


class Generator:
    def __init__(self):
        mp, model_path = ART / "meta.json", ART / "model.keras"
        if not (mp.exists() and model_path.exists()):
            raise FileNotFoundError("Model not trained yet. Run: python train.py")
        from tensorflow import keras
        self.meta = json.loads(mp.read_text())
        self.vocab, self.seq_len, self.seeds = self.meta["vocab"], self.meta["seq_len"], self.meta["seeds"]
        self.model = keras.models.load_model(model_path)

    @staticmethod
    def _sample(probs, temp, rng):
        logits = np.log(np.clip(np.asarray(probs, dtype="float64"), 1e-9, 1)) / temp
        p = np.exp(logits - logits.max())
        return int(rng.choice(len(p), p=p / p.sum()))

    def generate(self, length=64, temperature=1.0, bpm=100, seed=None):
        rng = np.random.default_rng(seed)
        window = list(self.seeds[int(rng.integers(len(self.seeds)))])
        out = []
        for _ in range(length):
            x = np.array([window[-self.seq_len:]], dtype="int32")
            probs = self.model(x, training=False).numpy()[0]
            t = temperature * (1.5 if len(out) >= 8 and len(set(out[-8:])) == 1 else 1.0)  # break loops
            i = self._sample(probs, t, rng)
            out.append(i)
            window.append(i)
        return self.to_notes(out)

    def to_notes(self, ids):
        notes, t = [], 0.0
        for i in ids:
            pitch_s, dur_s = self.vocab[i].rsplit("_", 1)
            pitches = [min(max(int(p), 0), 127) for p in pitch_s.split(".")]
            dur = float(dur_s)
            notes.append({"pitches": pitches, "start": t, "dur": dur})
            t += dur
        return notes

    @staticmethod
    def save_midi(notes, path, bpm=100):
        s = stream.Stream()
        s.insert(0, tempo.MetronomeMark(number=bpm))
        s.insert(0, instrument.Piano())
        for n in notes:
            el = chord.Chord(n["pitches"]) if len(n["pitches"]) > 1 else note.Note(n["pitches"][0])
            el.quarterLength = n["dur"]
            s.insert(n["start"], el)
        s.write("midi", fp=str(path))
