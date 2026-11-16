"""
PREPROCESSING HARBox -> sampel 900 dimensi (9 channel x 100 timestep)

Mengikuti data_pre.py bawaan dataset:
  - Struktur  : <root>/<user_id>/<Kelas>_train.txt
  - Tiap file : tabel waktu dengan 10 kolom, dibaca per blok 100 baris
                (1 blok = 1 window = 1 sampel)
  - Kolom 0 dibuang, kolom 1..9 = 9 channel IMU
  - Label     : Call=0, Hop=1, typing=2, Walk=3, Wave=4

Jalankan:
    python prep_harbox.py large_scale_HARBox
Output:
    harbox_windows.npz  (X: [N, 9, 100], y: [N], users: [N], classes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\User\Documents\ITB\TA\Code\CobaFlower\Dataset\HARBOX\large_scale_HARBox")
OUT = sys.argv[2] if len(sys.argv) > 2 else "harbox_windows.npz"

CLASS_SET = ["Call", "Hop", "typing", "Walk", "Wave"]  # label 0..4
W = 100          # baris per window (2 detik pada 50 Hz)
N_COLS = 10      # kolom per file
N_CH = 9         # channel IMU (kolom 1..9)

# Folder user = folder yang namanya angka (0, 1, 2, ...)
user_dirs = sorted(
    (d for d in ROOT.iterdir() if d.is_dir() and d.name.isdigit()),
    key=lambda d: int(d.name),
)
print(f"{len(user_dirs)} folder user di {ROOT}")

X, y, users = [], [], []
checked_col0 = False

for d in user_dirs:
    for label, cls in enumerate(CLASS_SET):
        f = d / f"{cls}_train.txt"
        if not f.exists():          # tidak semua user punya semua aktivitas
            continue

        arr = pd.read_csv(f, sep=r"\s+", header=None).to_numpy(dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != N_COLS:
            raise ValueError(f"{f}: bentuk {arr.shape}, diharapkan (*, {N_COLS})")

        # Info kolom 0 (dibuang) -- dicek sekali saja untuk dilihat manual
        if not checked_col0:
            print(f"Kolom 0 di {f.name}: 5 nilai awal = {arr[:5, 0]} "
                  f"(min={arr[:, 0].min():.3f}, max={arr[:, 0].max():.3f})")
            checked_col0 = True

        n = len(arr) // W
        if len(arr) % W:
            print(f"  peringatan: {f} punya {len(arr)} baris, sisa "
                  f"{len(arr) % W} baris dibuang")
        if n == 0:
            continue

        win = arr[: n * W].reshape(n, W, N_COLS)[:, :, 1:]   # (n, 100, 9)
        win = win.transpose(0, 2, 1)                          # (n, 9, 100)

        X.append(win)
        y.append(np.full(n, label, dtype=np.int64))
        users.append(np.full(n, d.name))

X = np.concatenate(X).astype(np.float32)
y = np.concatenate(y)
users = np.concatenate(users)

np.savez(OUT, X=X, y=y, users=users, classes=np.array(CLASS_SET))

# ----------------------------------------------------------------------
# Ringkasan untuk pengecekan manual
# ----------------------------------------------------------------------
uid, cnt = np.unique(users, return_counts=True)
n_cls_per_user = np.array([len(np.unique(y[users == u])) for u in uid])
print("\nX:", X.shape, "(N, channel, timestep)")
print("Per kelas :", dict(zip(CLASS_SET, np.bincount(y, minlength=5))))
print(f"Jumlah user: {len(uid)}")
print(f"Sampel per user: min={cnt.min()}  max={cnt.max()}  "
      f"rata-rata={cnt.mean():.1f}  std={cnt.std():.1f}")
print(f"User dengan kelima kelas lengkap: {(n_cls_per_user == 5).sum()} / {len(uid)}")
print(f"Tersimpan di {OUT}")