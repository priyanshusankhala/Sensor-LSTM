# 01_analyze.py
# import pandas as pd
# import numpy as np

# COLS = ["date", "time", "epoch", "moteid",
#         "temperature", "humidity", "light", "voltage"]

# # whitespace-separated; combine date+time into a timestamp
# df = pd.read_csv("/Users/priyanshu/Downloads/sensor/code/data/data.txt", sep=r"\s+", header=None, names=COLS,
#                  na_values=["", "NaN"])

# print("Shape:", df.shape)
# print("\n--- Data types ---")
# print(df.dtypes)
# print("\n--- Missing values per column ---")
# print(df.isna().sum())
# print("\n--- Basic stats ---")
# print(df[["temperature", "humidity", "light", "voltage"]].describe())
# print("\n--- moteid range ---")
# print(df["moteid"].min(), "to", df["moteid"].max(),
#       "| unique motes:", df["moteid"].nunique())

import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["temperature", "humidity", "light", "voltage"]
W, TRAIN_SPLIT = 25, 0.80

def load_clean_mote(path="/Users/priyanshu/Downloads/sensor/code/data/data.txt", mote=1, max_gap=5):
    cols = ["date","time","epoch","moteid",
            "temperature","humidity","light","voltage"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)

    # 1) DROP structurally-broken rows (garbage moteids, unparseable time)
    df["ts"] = pd.to_datetime(df.date + " " + df.time, errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df[(df.moteid >= 1) & (df.moteid <= 54)]          # schema range [[1]]

    # 2) select ONE mote -> clean single-node time series
    df = df[df.moteid == mote].sort_values("ts").reset_index(drop=True)
    print(f"Mote {mote}: {len(df)} rows before value cleaning")

    # 3) mask physically-impossible values (dying-battery garbage) -> NaN
    df.loc[~df.temperature.between(-10, 60),  "temperature"] = np.nan
    df.loc[~df.humidity.between(0, 100),      "humidity"]    = np.nan
    df.loc[~df.light.between(0, 200000),      "light"]       = np.nan   # [[1]]
    df.loc[~df.voltage.between(1.5, 3.5),     "voltage"]     = np.nan   # [[1]]

    # 4) report missing AFTER filtering to one mote
    print("Missing after mote-filter:\n", df[FEATURES].isna().sum())

    # 5) forward-fill SHORT gaps only (env data changes slowly) [[11]]
    df[FEATURES] = df[FEATURES].ffill(limit=max_gap)

    # 6) drop whatever remains (long gaps -> keep DPS sync honest)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"Mote {mote}: {len(df)} rows after cleaning")
    return df[FEATURES].to_numpy("float32")

data = load_clean_mote(mote=1)
split = int(len(data) * TRAIN_SPLIT)
scaler = MinMaxScaler().fit(data[:split])   # fit on TRAIN only [[11]]
scaled = scaler.transform(data)
np.save("scaled.npy", scaled)
