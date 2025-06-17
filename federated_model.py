import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import scipy.sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentClassifier(nn.Module):
    """Neural network architecture for binary sentiment classification."""

    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


class FederatedClient:
    """Client participating in federated learning."""

    def __init__(self, client_id, model, device='cpu'):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device

    def load_data(self, data_path):
        """Load and preprocess data for the client, handling sparse matrices."""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        def ensure_dense(x):
            return x.toarray() if scipy.sparse.issparse(x) else x

        X_train = ensure_dense(data['X_train']).astype(np.float32)
        X_test = ensure_dense(data['X_test']).astype(np.float32)

        self.X_train = torch.tensor(X_train).to(self.device)
        self.y_train = torch.tensor(data['y_train'].reshape(-1, 1)).float().to(self.device)
        self.X_test = torch.tensor(X_test).to(self.device)
        self.y_test = torch.tensor(data['y_test'].reshape(-1, 1)).float().to(self.device)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = TensorDataset(self.X_test, self.y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        print(f"Client {self.client_id} - Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")

    def train(self, epochs=5, learning_rate=0.001):
        """Train the client's model locally."""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_loss)

            if epoch % 2 == 0:
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return train_losses

    def evaluate(self):
        """Evaluate model performance on the local test set."""
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                outputs = self.model(batch_X)
                predicted = (outputs > 0.5).float()
                predictions.extend(predicted.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(actuals, predictions)
        return accuracy, predictions, actuals

    def get_parameters(self):
        """Return a list of the current model parameters."""
        return [param.data.clone() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters to the provided values."""
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = new_param.clone()


class FederatedServer:
    """Central server coordinating federated learning."""

    def __init__(self, model_template, num_clients):
        self.model_template = model_template
        self.num_clients = num_clients
        self.global_model = model_template
        self.round_accuracies = []

    def aggregate_parameters(self, client_parameters):
        """Aggregate model parameters from clients using FedAvg."""
        aggregated_params = []
        for param_idx in range(len(client_parameters[0])):
            stacked_params = torch.stack([client[param_idx] for client in client_parameters])
            avg_param = torch.mean(stacked_params, dim=0)
            aggregated_params.append(avg_param)
        return aggregated_params

    def update_global_model(self, aggregated_parameters):
        """Update global model with new parameters."""
        for param, new_param in zip(self.global_model.parameters(), aggregated_parameters):
            param.data = new_param.clone()

    def get_global_parameters(self):
        """Return current global model parameters."""
        return [param.data.clone() for param in self.global_model.parameters()]


def run_federated_learning(num_rounds=10, num_clients=5, epochs_per_round=3):
    """Run federated learning for specified number of rounds."""
    input_size = 3000  # Size of input features (e.g., TF-IDF vector size)
    global_model = SentimentClassifier(input_size)
    server = FederatedServer(global_model, num_clients)

    clients = []
    for i in range(num_clients):
        client_model = SentimentClassifier(input_size)
        client = FederatedClient(i, client_model)
        try:
            client.load_data(f'federated_data/client_{i}_data.pkl')
            clients.append(client)
        except FileNotFoundError:
            print(f"Data for client {i} not found. Skipping...")
            continue

    print(f"Starting federated learning with {len(clients)} clients for {num_rounds} rounds")

    round_results = []
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        global_params = server.get_global_parameters()

        client_parameters = []
        client_accuracies = []

        for client in clients:
            client.set_parameters(global_params)
            print(f"Training client {client.client_id}...")
            client.train(epochs=epochs_per_round)
            accuracy, _, _ = client.evaluate()
            client_accuracies.append(accuracy)
            print(f"Client {client.client_id} accuracy: {accuracy:.4f}")
            client_parameters.append(client.get_parameters())

        aggregated_params = server.aggregate_parameters(client_parameters)
        server.update_global_model(aggregated_params)

        avg_accuracy = np.mean(client_accuracies)
        server.round_accuracies.append(avg_accuracy)

        round_results.append({
            'round': round_num + 1,
            'avg_accuracy': avg_accuracy,
            'client_accuracies': client_accuracies
        })

        print(f"Round {round_num + 1} - Average accuracy: {avg_accuracy:.4f}")

    return server, clients, round_results


def plot_results(round_results):
    """Visualize the learning performance over rounds."""
    rounds = [r['round'] for r in round_results]
    avg_accuracies = [r['avg_accuracy'] for r in round_results]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, avg_accuracies, 'b-o', linewidth=2, markersize=6)
    plt.title('Federated Learning Performance')
    plt.xlabel('Round')
    plt.ylabel('Average Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    final_accuracies = round_results[-1]['client_accuracies']
    plt.bar(range(len(final_accuracies)), final_accuracies, alpha=0.7)
    plt.title('Final Round - Client Accuracies')
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('federated_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()