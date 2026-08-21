# task1_independent_vs_joint.py
import numpy as np, tensorflow as tf, joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

FEATURES = ["temperature", "humidity", "light", "voltage"]
S, W, EPOCHS = len(FEATURES), 25, 40
data = np.load("data_clean.npy")          # your 36,936 clean rows (save this in train.py)
split = int(len(data)*0.8)

def windows(a, W):
    return np.stack([a[i:i+W] for i in range(len(a)-W)]), a[W:]

def build(n_out):
    m = tf.keras.Sequential([tf.keras.layers.Input((W, n_out)),
                             tf.keras.layers.LSTM(32, unroll=True),
                             tf.keras.layers.Dense(n_out)])
    m.compile("adam", "mse"); return m

# ---------- JOINT (thesis) ----------
sc = MinMaxScaler().fit(data[:split]); sca = sc.transform(data)
Xtr, ytr = windows(sca[:split], W); Xva, yva = windows(sca[split:], W)
joint = build(S); joint.fit(Xtr, ytr, epochs=EPOCHS, batch_size=64, verbose=0)
pj = joint.predict(Xva, verbose=0)
print("JOINT overall R²:", round(r2_score(yva, pj), 3))

# ---------- INDEPENDENT (one LSTM per sensor) ----------
indep_pred = np.zeros_like(yva)
for i, f in enumerate(FEATURES):
    d = data[:, i:i+1]
    s = MinMaxScaler().fit(d[:split]); ds = s.transform(d)
    xtr, ytr1 = windows(ds[:split], W); xva, yva1 = windows(ds[split:], W)
    mi = build(1); mi.fit(xtr, ytr1, epochs=EPOCHS, batch_size=64, verbose=0)
    indep_pred[:, i] = mi.predict(xva, verbose=0).ravel()
    print(f"  INDEP {f} R²: {r2_score(yva1, indep_pred[:, i]):.3f}")

print("INDEP overall R²:", round(r2_score(yva, indep_pred), 3))
