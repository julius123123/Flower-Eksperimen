import sys
import numpy as np
import pandas as pd

MAX_GAP_NS = 1_000_000_000
W = 100
FS_MS = 20

df = pd.read_csv(r'C:\Users\User\Documents\ITB\TA\Code\CobaFlower\Dataset\HHAR\Activity recognition exp\Activity recognition exp\Phones_accelerometer.csv')

df = df.dropna(subset=["gt"])
df = df[df["gt"] != "null"]
df["Creation_Time"] = df["Creation_Time"].astype(np.int64)

classes = sorted(df["gt"].unique())
cls2id = {cls: i for i, cls in enumerate(classes)}
print("Classes:", classes)

X, y, users = [], [], []

for (user, device), g in df.groupby(["User", "Device"], sort = False):
    g = g.sort_values("Creation_Time")
    t = g["Creation_Time"].to_numpy()
    label = g["gt"].to_numpy()

    xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float32)

    # Segment the data that has differernt label or time gap larger than MAX_GAP_NS
    brk =  (label[1:] != label[:-1]) | (np.diff(t) > MAX_GAP_NS)
    # Contoh: b = [0, 500, 1200, 3000] -> segmen [0:500], [500:1200], ...
    b = np.r_[0, np.flatnonzero(brk) + 1, len(label)]

    for s, e in zip(b[:-1], b[1:]):
        if e - s < 2:
            continue

        seg = pd.DataFrame(xyz[s:e], columns=["x", "y", "z"],  index=pd.to_datetime(t[s:e], unit="ns"))

        seg = seg.resample(f"{FS_MS}ms").mean().interpolate(limit_direction="both")
        arr = seg.to_numpy(dtype=np.float32) 
        

        for i in range(0, len(arr) - W + 1, W):
            X.append(arr[i:i+W].T)
            y.append(cls2id[label[s]])
            users.append(user)

X = np.stack(X).astype(np.float32)
y = np.array(y, dtype=np.int64)
users = np.array(users)

np.savez("hhar_windows.npz", X=X, y=y, users=users, classes=np.array(classes))



# Ringkasan 
print("X:", X.shape, "(N, channel, timestep)")
print("Per kelas :", dict(zip(classes, np.bincount(y))))
print("Per user  :", dict(zip(*np.unique(users, return_counts=True))))
