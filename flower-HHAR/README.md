# Federated Learning with HHAR

Flower application for federated human activity recognition using the HHAR
dataset and a 1D CNN implemented with PyTorch.

## Requirements

- Python 3.12
- Flower with simulation support
- PyTorch
- NumPy
- pandas
- scikit-learn

## Setup

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Prepare the dataset

The preprocessing script reads the HHAR
`Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv`
file. Update the input path in `prep.py` if the dataset is stored elsewhere,
then run:

```powershell
python prep.py
```

This creates `hhar_windows.npz` containing:

- `X`: samples with shape `(N, 3, 100)`
- `y`: activity labels
- `users`: user identifier for each sample

The `.npz` file is intentionally ignored by Git. Keep it in this directory
when running the application.

In `pyproject.toml`, use a path accessible to the Flower runtime:

```toml
data-path = "hhar_windows.npz"
```

## Run federated training

```powershell
.\.venv\Scripts\Activate.ps1
flwr run . --stream
```

Training configuration is located in `pyproject.toml`, including the number of
rounds, learning rate, batch size, and model saving option.

## Outputs

When `save-model = true`, the final model is saved as:

```text
final_model.pt
```

Training metrics are saved under:

```text
outputs\<date>\<time>\results.json
```

## Plot results

```powershell
python graphic.py outputs\<date>\<time>\results.json
```

This creates `results.csv` and `results.png` in the same output directory.
