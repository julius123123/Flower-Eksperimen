"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy import FedAdagrad
from flwrCode.custom_strategy import CustomFedAdagrad

from datetime import datetime
from pathlib import Path
import json
from datetime import datetime

from flwrCode.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()



@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    data_path: str = context.run_config["data-path"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate) #fraction-train/ fraction-evaluate: select n% of client
    # strategy = FedAdagrad(fraction_evaluate=fraction_evaluate)
    # strategy = CustomFedAdagrad(fraction_evaluate=fraction_evaluate)


    # current_time = datetime.now()
    # run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # save_path = Path.cwd() / f"outputs/{run_dir}"
    # save_path.mkdir(parents=True, exist_ok=False)

    # strategy.set_save_path(save_path)
    # Start strategy, run FedAvg for `num_rounds`, then return result.
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arr: global_evaluate(server_round, arr, data_path)
    )

    out_dir = Path.cwd() / "outputs" / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_clientapp": {r: dict(m) for r, m in result.train_metrics_clientapp.items()},
        "eval_clientapp": {r: dict(m) for r, m in result.evaluate_metrics_clientapp.items()},
        "eval_global": {r: dict(m) for r, m in result.evaluate_metrics_serverapp.items()},
    }
    (out_dir / "results.json").write_text(json.dumps(history, indent=2))
    print(f"Hasil disimpan di {out_dir / 'results.json'}")

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord, data_path: str) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset(data_path)

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
