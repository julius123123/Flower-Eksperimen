"""pytorchexample: A Flower / PyTorch app."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


IN_CHANNELS = 3 # x, y, z
NUM_CLASSES = 6 # 6 labels

class Net(nn.Module):
    """CNN 1-D: 3 lapisan konv (feature extractor) + 2 FC (classifier)"""

    def __init__(self, in_ch = IN_CHANNELS, n_cls = NUM_CLASSES):
       super().__init__()
       self.features = nn.Sequential(
           nn.Conv1d(in_ch, 32, 5), nn.ReLU(), nn.MaxPool1d(2),
           nn.Conv1d(32, 64, 5), nn.ReLU(), nn.MaxPool1d(2),
           nn.Conv1d(64, 128, 3), nn.ReLU(),
           nn.AdaptiveAvgPool1d(1), nn.Flatten()
       )
       self.classifier = nn.Sequential(
           nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_cls)
       )

    def forward(self, x):
        return self.classifier(self.features(x))


# fds = None  # Cache FederatedDataset
_RAW = None
_CLIENTS = {}

# pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _resolve(data_path: str) -> Path:
    p = Path(data_path)
    candidates = [p, Path.cwd() / p, Path(__file__).resolve().parents[1] / p]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"'{data_path}' tidak ditemukan. Jalankan prep.py dulu, lalu isi "
        "data-path di pyproject.toml dengan path absolut ke hhar_windows.npz."
    )


def _load_raw(data_path):
    global _RAW
    if _RAW is None:
        d = np.load(_resolve(data_path))
        _RAW = (d["X"], d["y"], d["users"])

    return _RAW

def _client_arrays(user_idx, data_path):
    """Split 60/40 per user + normalisasi"""

    if user_idx in _CLIENTS:
        return _CLIENTS[user_idx]

    X, y, users = _load_raw(data_path)
    user_ids = sorted(np.unique(users))
    m = users == user_ids[user_idx] #user_idx=index
    Xc, yc = X[m], y[m]

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            Xc, yc, train_size=0.6, stratify=yc, random_state=42
        )
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            Xc, yc, train_size=0.6, random_state=42
        )


    mu = Xtr.mean((0, 2), keepdims=True)
    sd = Xtr.std((0, 2), keepdims=True) + 1e-8
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)

    _CLIENTS[user_idx] = (Xtr, Xte, ytr, yte)
    return _CLIENTS[user_idx]

def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# def apply_transforms(batch):
#     """Apply transforms to the partition from FederatedDataset."""
#     batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#     return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, data_path):
    """Data klien"""
    _, _, users = _load_raw(data_path)
    n_users = len(np.unique(users))
    if num_partitions != n_users:
        raise ValueError(
            f"num_supernodes={num_partitions} tetapi dataset ada {n_users} users"
        )

    Xtr, Xte, ytr, yte = _client_arrays(partition_id, data_path)
    return _loader(Xtr, ytr, batch_size, True), _loader(Xte, yte, batch_size, False)


def load_centralized_dataset(data_path, batch_size=128):
    """Load test set and return dataloader."""
    # Load entire test set
    _, _, users = _load_raw(data_path=data_path)
    parts = [_client_arrays(i, data_path) for i in range(len(np.unique(users)))]
    Xte = np.concatenate([p[1] for p in parts])
    yte = np.concatenate([p[3] for p in parts])
    return _loader(Xte, yte, batch_size, False)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for xb, yb in testloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = net(xb)
            loss += criterion(outputs, yb).item()
            correct += (outputs.argmax(1) == yb).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
