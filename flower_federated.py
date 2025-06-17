import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import scipy.sparse

# Import our custom model
from federated_model import SentimentClassifier


class FederatedClient:
    """Federated client to load data, train, and evaluate locally."""

    def __init__(self, client_id: int, model: nn.Module):
        self.client_id = client_id
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self, data_path: str):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        def ensure_dense(x):
            return x.toarray() if scipy.sparse.issparse(x) else x

        X_train = ensure_dense(data["X_train"]).astype(np.float32)
        X_test = ensure_dense(data["X_test"]).astype(np.float32)

        self.X_train = torch.tensor(X_train).to(self.device)
        self.y_train = torch.tensor(data["y_train"]).long().to(self.device)
        self.X_test = torch.tensor(X_test).to(self.device)
        self.y_test = torch.tensor(data["y_test"]).long().to(self.device)

    def train(self, epochs: int = 3, batch_size: int = 32, lr: float = 0.001):
        self.model.to(self.device)
        self.model.train()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        losses = []
        for epoch in range(epochs):
            permutation = torch.randperm(self.X_train.size(0))
            epoch_loss = 0.0
            for i in range(0, self.X_train.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.X_train[indices], self.y_train[indices].float()

                optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            losses.append(epoch_loss / ((i + batch_size) // batch_size))
        return losses

    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test).squeeze()
            predictions = (outputs >= 0.5).long()
            correct = (predictions == self.y_test).sum().item()
            total = self.y_test.size(0)
            accuracy = correct / total
        return accuracy, correct, total


class FlowerClient(fl.client.NumPyClient):
    """Flower client implementation."""

    def __init__(self, client_id: int, model: nn.Module, data_path: str):
        self.client_id = client_id
        self.model = model
        self.federated_client = FederatedClient(client_id, model)
        self.federated_client.load_data(data_path)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config.get("epochs", 3)
        losses = self.federated_client.train(epochs=epochs)
        return self.get_parameters({}), len(self.federated_client.X_train), {
            "client_id": self.client_id,
            "train_loss": losses[-1] if losses else 0.0,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy, _, _ = self.federated_client.evaluate()
        return float(accuracy), len(self.federated_client.X_test), {
            "client_id": self.client_id,
            "accuracy": accuracy,
        }


def create_flower_client(client_id: int) -> FlowerClient:
    model = SentimentClassifier(input_size=3000)
    data_path = f"federated_data/client_{client_id}_data.pkl"
    return FlowerClient(client_id, model, data_path)


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_accuracies = []

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        accuracies = [r.metrics["accuracy"] for _, r in results]
        avg_accuracy = np.mean(accuracies)
        self.round_accuracies.append(avg_accuracy)

        print(f"Round {server_round} - Average accuracy: {avg_accuracy:.4f}")
        return super().aggregate_evaluate(server_round, results, failures)


def run_flower_simulation(num_clients: int = 5, num_rounds: int = 10):
    def client_fn(cid: str) -> FlowerClient:
        return create_flower_client(int(cid))

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda server_round: {"epochs": 3},
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    return history, strategy


if __name__ == "__main__":
    print("Starting Flower federated learning simulation...")
    history, strategy = run_flower_simulation(num_clients=5, num_rounds=10)
    print("\nFederated learning completed!")
    print("Round accuracies:", strategy.round_accuracies)