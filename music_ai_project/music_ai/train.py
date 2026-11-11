"""Step 3 + 4: build and train the LSTM.   Usage: python train.py --epochs 20 --max-files 60"""
import argparse
from pathlib import Path
from data import ART, build_dataset


def build_model(vocab_size, seq_len):
    from tensorflow import keras
    L = keras.layers
    model = keras.Sequential([
        L.Input(shape=(seq_len,)),
        L.Embedding(vocab_size, 64),
        L.LSTM(256, return_sequences=True),
        L.Dropout(0.3),
        L.LSTM(256),
        L.Dropout(0.3),
        L.Dense(256, activation="relu"),
        L.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-dir", default="data/midi")
    ap.add_argument("--max-files", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()

    from tensorflow import keras
    X, y, meta = build_dataset(a.midi_dir, a.max_files, a.seq_len)
    model = build_model(len(meta["vocab"]), a.seq_len)
    model.summary()
    cb = [
        keras.callbacks.ModelCheckpoint(str(ART / "model.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    model.fit(X, y, epochs=a.epochs, batch_size=a.batch_size, validation_split=0.1, shuffle=True, callbacks=cb)
    print("Saved best model to", ART / "model.keras")
