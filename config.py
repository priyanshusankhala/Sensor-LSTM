# Parameters taken directly from the thesis
WINDOW_ENV = 25          # optimal window, environmental dataset (Sec 4.3)
WINDOW_MECH = 40         # optimal window, mechanical dataset (Sec 4.3)
LSTM_UNITS = 32          # single LSTM layer of 32 units (Sec 3.3.2)
TRAIN_SPLIT = 0.80       # 80/20 split, temporal order preserved (Sec 3.3.1)

# Compression schemes — environmental (Table 4.2). Winner = Scheme 4
ENV_SCHEMES = {
    1: 100, 2: 50, 3: "mixed", 4: 10, 5: 1
}
# Mechanical scaling (Table 4.3): (orientation, magnetometer). Winner = Scheme 2
MECH_SCHEMES = {
    1: (1000, 100), 2: (1000, 10), 3: (500, 10), 4: (500, 5), 5: (100, 5)
}

# Bit-packing profiles: k=8 for env (3-bit index), k=16 for mech (4-bit index) (Sec 4.5)
PROFILE_INDEX_BITS_ENV = 3
PROFILE_INDEX_BITS_MECH = 4

# LoRa energy/latency fitted models (Eqs. 5.3 & 5.5)
LORA_E0, LORA_S = 17.061, 0.695436     # Etx(B) = 17.061 + 0.695436*B  [mJ]
LORA_T0, LORA_ALPHA = 39.244, 1.469595 # Ttx(B) = 39.244 + 1.469595*B  [ms]
LSTM_ENERGY_MJ = 10.39                 # LSTM overhead @ W=25 (Table 5.2)
LSTM_LATENCY_MS = 15.174
