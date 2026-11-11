"""Flask backend.   Run: python app.py  ->  http://127.0.0.1:5000"""
import re, threading, uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from generate import Generator

app = Flask(__name__, static_folder="static")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
lock = threading.Lock()  # one generation at a time (Keras models are not thread-safe)
gen, load_error = None, None
try:
    gen = Generator()
except Exception as e:
    load_error = str(e)


def num(data, key, lo, hi, default, cast):
    v = data.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return v


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    return jsonify(ready=gen is not None, message=load_error)


@app.post("/api/generate")
def generate():
    if gen is None:
        return jsonify(error=load_error or "Model not loaded."), 503
    data = request.get_json(silent=True) or {}
    try:
        length = num(data, "length", 8, 400, 64, int)
        temp = num(data, "temperature", 0.2, 2.0, 1.0, float)
        bpm = num(data, "bpm", 40, 220, 100, int)
        seed = data.get("seed")
        seed = None if seed in (None, "") else num(data, "seed", 0, 2**31 - 1, 0, int)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    with lock:
        notes = gen.generate(length, temp, bpm, seed)
        gid = uuid.uuid4().hex[:12]
        Generator.save_midi(notes, OUT / f"{gid}.mid", bpm)
        for old in sorted(OUT.glob("*.mid"), key=lambda p: p.stat().st_mtime)[:-30]:  # keep latest 30
            old.unlink(missing_ok=True)
    return jsonify(id=gid, bpm=bpm, notes=notes, duration_beats=notes[-1]["start"] + notes[-1]["dur"])


@app.get("/api/download/<gid>")
def download(gid):
    path = OUT / f"{gid}.mid"
    if not re.fullmatch(r"[0-9a-f]{12}", gid) or not path.exists():
        return jsonify(error="File not found (old files are cleaned up automatically)."), 404
    return send_file(path, as_attachment=True, download_name=f"ai-music-{gid}.mid", mimetype="audio/midi")


@app.errorhandler(Exception)
def on_error(e):
    code = getattr(e, "code", 500)
    return jsonify(error=str(e) if code != 500 else "Something went wrong while generating. Try again."), code


if __name__ == "__main__":
    app.run(debug=False, port=5000)
