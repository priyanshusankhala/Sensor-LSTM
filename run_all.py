# run_all.py — master entry point
from src.window_search import find_best_window   # reproduces Figs 4.3/4.4
from src.evaluate import compare_schemes         # reproduces Figs 4.5–4.9
from src.lora_energy import savings

find_best_window("data/environmental.csv")       # expect W≈25
compare_schemes("data/environmental.csv")        # expect Scheme 4 optimal
print("SF11 energy savings:", round(savings(sf=11), 1), "%")  # ≈54.6%
