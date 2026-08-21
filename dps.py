# dps.py — Dual Prediction Scheme (CLOSED-LOOP, thesis-faithful)
# Chain: residual -> delta -> scale+round -> zig-zag -> adaptive bit-pack. [[11]]
# Both TX and RX advance their context window with the SAME reconstructed value,
# so they keep "identical LSTM states and context windows" (Sec 3.4.1). [[11]]

import numpy as np
import tensorflow as tf
import joblib

# ---------------- config (thesis parameters) ----------------
FEATURES   = ["temperature", "humidity", "light", "voltage"]
S          = len(FEATURES)
W          = 25          # environmental window [[11]]
N_ITERS    = 1000        # DPS iterations to simulate
SCALE      = 10          # Scheme 4 (x10): best compression/accuracy for env [[11]]
INDEX_BITS = 3           # k=8 profiles -> 3-bit profile index (environmental) [[11]]
RAW_BYTES  = 24          # actual environmental payload size = 24 bytes [[11]]

# ---------------- load trained artefacts (from train.py) ----------------
model  = tf.keras.models.load_model("dps_lstm.keras")
scaled = np.load("scaled.npy")           # Min-Max scaled series
scaler = joblib.load("scaler.pkl")       # to move between normalized <-> real units

# ---------------- compression helpers ----------------
def zigzag(n):                           # Table 3.3: small |values| stay small [[11]]
    n = int(n)
    return (n << 1) ^ (n >> 31)

def min_bits(u):                         # minimum bits for an unsigned int
    return max(1, int(u).bit_length())

def payload_bytes(q):
    """zig-zag + adaptive bit count + 3-bit profile index -> byte count. [[11]]"""
    zz = [zigzag(v) for v in q]
    total_bits = sum(min_bits(v) for v in zz) + INDEX_BITS   # + profile index [[11]]
    return int(np.ceil(total_bits / 8))                       # byte alignment [[11]]

def chan_bits(q):
    return np.array([min_bits(zigzag(int(v))) for v in q])

# ---------------- fast inference (avoid slow model.predict in a loop) ----------------
@tf.function(reduce_retracing=True)
def _predict(win):
    return model(win, training=False)

def infer(window_norm):
    """window_norm: (W,S) normalized -> returns REAL-scale prediction (S,)."""
    pred_norm = _predict(window_norm[None].astype("float32")).numpy()[0]
    return scaler.inverse_transform(pred_norm[None])[0]

# ---------------- Dual Prediction Scheme (closed loop) ----------------
# Both nodes start with the SAME context window -> perfect initial sync. [[11]]
window   = scaled[:W].copy()             # SHARED window (identical on TX & RX)
r_hat    = np.zeros(S)                    # previous RECONSTRUCTED residual (both sides)

comp_sizes, bits_log = [], []
actuals, recons = [], []

for t in range(W, W + N_ITERS):
    actual_real = scaler.inverse_transform(scaled[t][None])[0]   # true reading (at TX)

    # ---- both nodes predict IDENTICALLY (same window) ----
    pred_real = infer(window)

    # ---- TRANSMITTER: residual (real scale) -> delta vs last reconstructed residual ----
    residual = actual_real - pred_real            # residual, actual scale [[11]]
    delta    = residual - r_hat                   # delta of residual [[11]]
    q        = np.round(delta * SCALE).astype(int)  # scale + round (only lossy step) [[11]]
    # ^^^ ONLY q is "transmitted" (after zig-zag + adaptive bit-pack) ^^^

    # ---- reconstruction (identical on TX and RX, since same pred + same q) ----
    delta_hat  = q / SCALE
    r_hat      = r_hat + delta_hat                # updated reconstructed residual [[11]]
    recon_real = pred_real + r_hat                # X' = prediction + residual (Fig 3.13) [[11]]

    # ---- CLOSED LOOP: advance the SHARED window with the reconstructed value ----
    recon_norm = scaler.transform(recon_real[None])[0]
    window = np.vstack([window[1:], recon_norm])  # TX & RX stay perfectly in sync [[11]]

    # ---- logging ----
    comp_sizes.append(payload_bytes(q))
    bits_log.append(chan_bits(q))
    actuals.append(actual_real)
    recons.append(recon_real)

# ---------------- metrics (real units) ----------------
A = np.array(actuals); R = np.array(recons)
bits_log = np.array(bits_log)

mae  = np.mean(np.abs(A - R))
mape = np.mean(np.abs((A - R) / np.where(np.abs(A) < 1e-9, 1e-9, A))) * 100
comp = 100 * (1 - np.mean(comp_sizes) / RAW_BYTES)

print(f"\n===== DPS (closed-loop) over {N_ITERS} iterations, Scheme x{SCALE} =====")
print(f"Avg compressed payload : {np.mean(comp_sizes):.2f} B   (raw = {RAW_BYTES} B)")
print(f"Overall compression    : {comp:.1f} %   (thesis env target ~80%) [[11]]")
print(f"MAE  (real units)      : {mae:.4f}")
print(f"MAPE                   : {mape:.3f} %   (thesis env target <1%) [[11]]")

print("\n--- Per-channel ---")
for i, f in enumerate(FEATURES):
    ch_mae  = np.mean(np.abs(A[:, i] - R[:, i]))
    ch_mape = np.mean(np.abs((A[:, i] - R[:, i]) /
                             np.where(np.abs(A[:, i]) < 1e-9, 1e-9, A[:, i]))) * 100
    print(f"  {f:<12} avg bits: {bits_log[:, i].mean():5.2f} | "
          f"MAE: {ch_mae:8.4f} | MAPE: {ch_mape:7.3f} %")
