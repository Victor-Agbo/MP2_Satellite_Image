# server.py
import flwr as fl
import torch
from train_utils import build_model, set_weights, get_device
from flwr.common import parameters_to_ndarrays 

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, patience=5, **kwargs):
        super().__init__(**kwargs)
        self.best_f1_micro = 0.0
        self.patience = patience
        self.counter = 0

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        self.latest_parameters = aggregated_parameters 
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        if ndarrays is not None:
            model = build_model().to(get_device())
            set_weights(model, ndarrays)
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # Call FedAvg's default aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Collect metrics from all clients
        f1_micros = []
        f1_macros = []
        val_losses = []

        for _, evaluate_res in results:
            metrics = evaluate_res.metrics  # this is the dict
            if "f1_micro" in metrics:
                f1_micros.append(metrics["f1_micro"])
            if "f1_macro" in metrics:
                f1_macros.append(metrics["f1_macro"])
            val_losses.append(evaluate_res.loss)  # loss is separate


        # Compute averages
        avg_f1_micro = sum(f1_micros) / len(f1_micros) if f1_micros else None
        avg_f1_macro = sum(f1_macros) / len(f1_macros) if f1_macros else None
        avg_val_loss = aggregated_loss

        # Print aggregated metrics
        print(f"📊 Round {server_round} aggregated metrics:")
        print(f"   val_loss={avg_val_loss:.4f}, f1_micro={avg_f1_micro:.4f}, f1_macro={avg_f1_macro:.4f}")

        # --- Early stopping based on F1-micro ---
        if avg_f1_micro is not None:
            if avg_f1_micro > self.best_f1_micro:
                self.best_f1_micro = avg_f1_micro
                self.counter = 0
                print(f"💾 New best F1-micro={avg_f1_micro:.4f} at round {server_round}")
                
                # Save best global model checkpoint using latest aggregated parameters
                ndarrays = parameters_to_ndarrays(self.latest_parameters)
                model = build_model().to(get_device())
                set_weights(model, ndarrays)
                torch.save({"model_state_dict": model.state_dict()},
                        f"best_fed_model.pth")
                print(f"✅ Saved best global model at round {server_round}")

            else:
                self.counter += 1
                print(f"⚠️ No improvement in F1-micro. Patience {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    print(f"⏹️ Early stopping triggered at round {server_round}")
                    raise SystemExit("Early stopping")


        # Return aggregated loss and metrics to Flower
        return aggregated_loss, {
            "f1_micro": avg_f1_micro,
            "f1_macro": avg_f1_macro,
            "val_loss": avg_val_loss,
        }
def main():
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=100)
    )

if __name__ == "__main__":
    main()
