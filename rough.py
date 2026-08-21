import numpy as np
d = np.load("data_clean.npy")
print("Shape:", d.shape)              # (36936, 4)
print("Temp range:", d[:,0].min(), "to", d[:,0].max())  # real °C values, e.g. ~19–24
