"""Ubah results.json (output server_app.py) menjadi CSV dan grafik.

Pakai:
    python plot_results.py path/ke/results.json

Output (di folder yang sama dengan results.json):
    results.csv   -> satu baris per ronde
    results.png   -> grafik akurasi dan loss
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # simpan ke file, tidak perlu jendela GUI
import matplotlib.pyplot as plt
import pandas as pd

src = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\User\Documents\ITB\TA\Code\CobaFlower\flower-HHAR\outputs\2026-11-15\21-20-35\results.json")
out_dir = src.parent
history = json.loads(src.read_text())

# ----------------------------------------------------------------------
# 1. JSON -> tabel (index = nomor ronde, kolom = "<bagian>.<metrik>")
# ----------------------------------------------------------------------
df = pd.concat(
    {name: pd.DataFrame.from_dict(d, orient="index") for name, d in history.items() if d},
    axis=1,
)
df.index = df.index.astype(int)  # kunci JSON berupa string -> ubah ke angka
df = df.sort_index()
df.columns = [f"{a}.{b}" for a, b in df.columns]

csv_path = out_dir / "results.csv"
df.to_csv(csv_path, index_label="round")
print(f"CSV   : {csv_path}  ({len(df)} baris, kolom: {list(df.columns)})")

# ----------------------------------------------------------------------
# 2. Grafik: kiri = akurasi, kanan = loss
# ----------------------------------------------------------------------
ACC = {
    "eval_clientapp.eval_acc": "Eval klien (rata-rata 9 user)",
    "eval_global.accuracy": "Eval global (gabungan test)",
}
LOSS = {
    "train_clientapp.train_loss": "Train loss (klien)",
    "eval_clientapp.eval_loss": "Eval loss (klien)",
    "eval_global.loss": "Eval loss (global)",
}


def draw(ax, cols, title, ylabel):
    for col, label in cols.items():
        if col in df.columns:
            s = df[col].dropna()
            ax.plot(s.index, s.values, marker="o", markersize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel("Ronde FL")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
draw(ax1, ACC, "Akurasi", "Akurasi")
draw(ax2, LOSS, "Loss", "Loss")
fig.tight_layout()

png_path = out_dir / "results.png"
fig.savefig(png_path, dpi=150)
print(f"Grafik: {png_path}")

# ----------------------------------------------------------------------
# 3. Ringkasan di terminal
# ----------------------------------------------------------------------
col = "eval_clientapp.eval_acc"
if col in df.columns:
    s = df[col].dropna()
    print(f"\nAkurasi eval klien: ronde terakhir = {s.iloc[-1]:.4f} "
          f"(ronde {s.index[-1]}), terbaik = {s.max():.4f} (ronde {s.idxmax()})")
