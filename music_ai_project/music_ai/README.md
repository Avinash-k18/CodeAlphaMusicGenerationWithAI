# Melody Lab – Music Generation with AI (LSTM)

## Pipeline (matches Task 3)
1. **Collect MIDI** – put `.mid` files in `data/midi/` (classical, jazz…). If the folder is empty, music21's built-in Bach corpus is used automatically.
2. **Preprocess** (`data.py`) – music21 parses each file, takes the busiest track, converts notes/chords to tokens (`pitch_duration`), drops rare tokens, builds sliding windows of 32 tokens -> next token.
3. **Model** (`train.py`) – Embedding -> 2x LSTM(256) + Dropout -> Dense -> Softmax.
4. **Train** – early stopping, LR reduction, best model saved to `artifacts/model.keras`.
5. **Generate -> MIDI** (`generate.py`) – temperature sampling, saved as `.mid`; the web UI plays it in the browser and lets you download it.

## Run
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py --max-files 60 --epochs 20          # ~10-30 min on CPU; use Google Colab for speed
python app.py                                       # open http://127.0.0.1:5000
```
Quick test: `python train.py --max-files 15 --epochs 5`.

## Edge cases handled
Corrupt/unsupported files skipped · too-short pieces skipped · no MIDI folder -> built-in corpus · tiny dataset/vocab -> clear error ·
rare notes removed · chords and tuplet/zero durations quantized · repetition loops broken during sampling ·
model missing -> UI shows instructions, 503 from API · input validated (length, temperature, tempo, seed) -> 400 ·
concurrent requests serialized · download id sanitized (no path traversal) · old outputs auto-deleted · browser audio blocked/unsupported message.

## Limits (be honest in your report)
Models one melody line (not full polyphony); output is piano-like synth in browser, not a rendered audio file (convert `.mid` with FluidSynth/MuseScore if needed).
