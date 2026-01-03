# server.py
import flwr as fl
import torch
from train_utils import build_model, set_weights, get_device

# def main():
#     # Define the federated averaging strategy
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,          # use all clients for training
#         fraction_evaluate=1.0,     # use all clients for evaluation
#         min_fit_clients=8,         # expect 8 clients to fit
#         min_evaluate_clients=8,    # expect 8 clients to evaluate
#         min_available_clients=8    # require 8 clients connected
#     )

#     # Start the server
#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         strategy=strategy,
#         config=fl.server.ServerConfig(num_rounds=2)  # number of federated rounds
#     )

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call FedAvg's aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert Flower parameters to PyTorch model
            model = build_model().to(get_device())
            set_weights(model, aggregated_parameters)

            # Save the global model after each round
            torch.save({"model_state_dict": model.state_dict()},
                    f"best_fed_round_{server_round}.pth")
            print(f"✅ Saved global model at round {server_round}")

        return aggregated_parameters, aggregated_metrics


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
        config=fl.server.ServerConfig(num_rounds=2)
    )

if __name__ == "__main__":
    main()
