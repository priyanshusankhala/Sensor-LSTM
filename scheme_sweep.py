# scheme_sweep.py — reproduce thesis Figs 4.5a, 4.6a, 4.9a (environmental schemes) [[11]]
import numpy as np, tensorflow as tf, joblib
import matplotlib.pyplot as plt

FEATURES = ["temperature", "humidity", "light", "voltage"]
S, W, N_ITERS, INDEX_BITS, RAW_BYTES = len(FEATURES), 25, 1000, 3, 24

# Thesis environmental schemes (Table 4.2): Scheme 3 = mixed. [[11]]
# Here: index 2 (LUX-like) gets x100, rest x10 to mimic the "mixed" scheme.
SCHEMES = {1: 100, 2: 50, 3: "mixed", 4: 10, 5: 1}

model  = tf.keras.models.load_model("dps_lstm.keras")
scaled = np.load("scaled.npy")
scaler = joblib.load("scaler.pkl")

def zigzag(n): n = int(n); return (n << 1) ^ (n >> 31)
def min_bits(u): return max(1, int(u).bit_length())

@tf.function(reduce_retracing=True)
def _p(w): return model(w, training=False)
def infer(w): return scaler.inverse_transform(_p(w[None].astype("float32")).numpy())[0]

def scale_vec(scheme):
    if scheme == "mixed":                     # Scheme 3: light x100, rest x10 [[11]]
        v = np.full(S, 10.0); v[FEATURES.index("light")] = 100.0; return v
    return np.full(S, float(scheme))

def run_scheme(scheme):
    sf = scale_vec(scheme)
    window = scaled[:W].copy(); r_hat = np.zeros(S)
    sizes, A, R = [], [], []
    for t in range(W, W + N_ITERS):
        actual = scaler.inverse_transform(scaled[t][None])[0]
        pred   = infer(window)
        delta  = (actual - pred) - r_hat
        q      = np.round(delta * sf).astype(int)          # per-channel scaling [[11]]
        zz     = [zigzag(v) for v in q]
        sizes.append(int(np.ceil((sum(min_bits(v) for v in zz) + INDEX_BITS) / 8)))
        r_hat  = r_hat + q / sf
        recon  = pred + r_hat
        window = np.vstack([window[1:], scaler.transform(recon[None])[0]])
        A.append(actual); R.append(recon)
    A, R = np.array(A), np.array(R)
    comp = 100 * (1 - np.mean(sizes) / RAW_BYTES)
    mae  = np.mean(np.abs(A - R))
    mape = np.mean(np.abs((A - R) / np.where(np.abs(A) < 1e-9, 1e-9, A))) * 100
    return comp, mae, mape

comps, maes, mapes = [], [], []
for s in SCHEMES:
    c, m, p = run_scheme(SCHEMES[s])
    comps.append(c); maes.append(m); mapes.append(p)
    print(f"Scheme {s}: compression={c:5.1f}%  MAE={m:.4f}  MAPE={p:.3f}%")

# --- plots (mirror Figs 4.5a, 4.6a, 4.9a) ---
x = list(SCHEMES.keys())
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].bar(x, comps); ax[0].set_title("Compression % (Fig 4.5a)"); ax[0].set_xlabel("Scheme")
ax[1].bar(x, maes);  ax[1].set_title("MAE (Fig 4.6a)");           ax[1].set_xlabel("Scheme")
ax[2].bar(x, mapes); ax[2].axhline(1.0, ls="--", c="r", label="1% target")
ax[2].set_title("Overall MAPE % (Fig 4.9a)"); ax[2].set_xlabel("Scheme"); ax[2].legend()
plt.tight_layout(); plt.savefig("scheme_sweep.png", dpi=150)
print("Saved scheme_sweep.png")
