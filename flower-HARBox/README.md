# Federated Learning with HARBox

Flower application for federated activity recognition using the HARBox dataset
and a 1D CNN implemented with PyTorch.

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

The raw dataset is expected to contain user directories with files such as:

```text
<dataset-root>\<user-id>\Call_train.txt
<dataset-root>\<user-id>\Hop_train.txt
<dataset-root>\<user-id>\typing_train.txt
<dataset-root>\<user-id>\Walk_train.txt
<dataset-root>\<user-id>\Wave_train.txt
```

Generate the local `.npz` file:

```powershell
python data_prep.py C:\path\to\large_scale_HARBox harbox_windows.npz
```

The generated file contains:

- `X`: samples with shape `(N, 9, 100)`
- `y`: activity labels for five classes
- `users`: user identifier for each sample

The `.npz` file is intentionally ignored by Git. Keep it in this directory
when running the application.

In `pyproject.toml`, use a path accessible to the Flower runtime:

```toml
data-path = "harbox_windows.npz"
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
