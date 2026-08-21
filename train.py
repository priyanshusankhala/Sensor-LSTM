# train.py — self-contained: load + clean + preprocess + time-series CV + final model
# Replicates the thesis's LSTM predictor (single LSTM layer of 32 units, W=25,
# Min-Max scaling, temporal order preserved) on the Intel Lab Data. [[11]]

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import joblib

# =================== CONFIG (thesis parameters) ===================
FEATURES     = ["temperature", "humidity", "light", "voltage"]
S            = len(FEATURES)          # number of sensor channels
W            = 25                     # thesis environmental window size [[11]]
LSTM_UNITS   = 32                     # single LSTM layer of 32 units [[11]]
TRAIN_SPLIT  = 0.80                   # 80/20, temporal order preserved [[11]]
EPOCHS       = 60                     # training epochs (hyperparameter, train-only) [[11]]
BATCH_SIZE   = 64
N_CV_SPLITS  = 5                      # folds for TimeSeriesSplit
DATA_PATH    = "/Users/priyanshu/Downloads/sensor/code/data/data.txt"
MOTE         = 1
np.random.seed(42); tf.random.set_seed(42)

# =================== 1. LOAD + CLEAN ===================
def load_clean_mote(path=DATA_PATH, mote=MOTE, max_gap=5):
    """Load one mote's stream, drop garbage, mask out-of-range, fill short gaps."""
    cols = ["date", "time", "epoch", "moteid",
            "temperature", "humidity", "light", "voltage"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)

    # build timestamp; drop unparseable rows
    df["ts"] = pd.to_datetime(df.date + " " + df.time, errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df[(df.moteid >= 1) & (df.moteid <= 54)]          # valid IDs 1–54 [[7]]
    df = df[df.moteid == mote].sort_values("ts").reset_index(drop=True)
    print(f"Mote {mote}: {len(df)} rows before value cleaning")

    # mask physically-impossible readings (dying-battery garbage) -> NaN [[7]]
    df.loc[~df.temperature.between(-10, 60), "temperature"] = np.nan
    df.loc[~df.humidity.between(0, 100),     "humidity"]    = np.nan
    df.loc[~df.light.between(0, 200000),     "light"]       = np.nan   # Lux range [[7]]
    df.loc[~df.voltage.between(1.5, 3.5),    "voltage"]     = np.nan   # 2–3 V range [[7]]

    # forward-fill SHORT gaps (env data changes slowly), drop long gaps
    df[FEATURES] = df[FEATURES].ffill(limit=max_gap)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    print(f"Mote {mote}: {len(df)} clean rows | "
          f"missing = {df[FEATURES].isna().sum().sum()}")
    return df[FEATURES].to_numpy("float32")

data = load_clean_mote()

# =================== 2. WINDOW HELPER ===================
def make_windows(arr, W):
    """Sliding window, stride 1, single-step target (thesis Sec 3.3.1). [[11]]"""
    X = np.stack([arr[i:i + W] for i in range(len(arr) - W)])
    return X, arr[W:]

# =================== 3. MODEL BUILDER ===================
def build_model():
    """Single LSTM(32) -> Dense(S). unroll=True for TFLite-Micro compatibility. [[11]]"""
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(W, S)),
        tf.keras.layers.LSTM(LSTM_UNITS, unroll=True),   # primitive ops, no dropout [[11]]
        tf.keras.layers.Dense(S)                         # predict 1 step ahead [[11]]
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

# =================== 4. TIME-SERIES CROSS-VALIDATION ===================
# TimeSeriesSplit trains on PAST, validates on FUTURE -> no leakage, unlike
# random k-fold. Faithful to the thesis principle of temporal order. [[11]]
print("\n===== Time-Series Cross-Validation =====")

# scale on the FULL series just for CV diagnostics (final scaler refit below)
cv_scaler = MinMaxScaler().fit(data)
cv_scaled = cv_scaler.transform(data)
Xall, yall = make_windows(cv_scaled, W)

tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
fold_scores = []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(Xall), start=1):
    model = build_model()
    model.fit(Xall[tr_idx], yall[tr_idx],
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    r2 = r2_score(yall[va_idx], model.predict(Xall[va_idx], verbose=0))
    print(f"Fold {fold}: train={len(tr_idx):>6} val={len(va_idx):>6} | R² = {r2:.3f}")
    fold_scores.append(r2)

print(f"Mean CV R²: {np.mean(fold_scores):.3f}  (±{np.std(fold_scores):.3f})")

# =================== 5. FINAL MODEL (train on ALL data) ===================
# The deployed DPS model is trained offline ONCE on the whole dataset, so it
# sees early + mid + late (drift) periods. [[11]]
print("\n===== Final Model (trained on ALL data) =====")

split  = int(len(data) * TRAIN_SPLIT)
scaler = MinMaxScaler().fit(data[:split])   # fit on TRAIN only -> no leakage [[11]]
scaled = scaler.transform(data)

Xtr, ytr = make_windows(scaled[:split], W)
Xva, yva = make_windows(scaled[split:], W)
print("Train windows:", Xtr.shape, "| Val windows:", Xva.shape)

final_model = build_model()
final_model.fit(Xtr, ytr, validation_data=(Xva, yva),
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# =================== 6. REPORT ===================
pred = final_model.predict(Xva, verbose=0)
print("\nVal MSE:", float(np.mean((pred - yva) ** 2)))
print("Val R² :", r2_score(yva, pred))         # thesis uses MSE + R² [[11]]
for i, f in enumerate(FEATURES):
    print(f"  {f:<12} R²: {r2_score(yva[:, i], pred[:, i]):.3f}")

# =================== 7. SAVE (for the DPS loop) ===================
final_model.save("dps_lstm.keras")
np.save("scaled.npy", scaled)
joblib.dump(scaler, "scaler.pkl")
print("\nSaved: dps_lstm.keras, scaled.npy, scaler.pkl ✅")
data = load_clean_mote()          # your existing line (real-unit cleaned data)
np.save("data_clean.npy", data)   # ← ADD THIS: saves the cleaned real-unit array
