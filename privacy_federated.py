import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from federated_model import FederatedClient


class PrivateFederatedClient(FederatedClient):
    """Federated client with differential privacy using Opacus."""

    def __init__(self, client_id, model, device='cpu', epsilon=1.0, delta=1e-5):
        super().__init__(client_id, model)
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_engine = None

    def train_with_privacy(self, epochs=5, learning_rate=0.001, max_grad_norm=1.0, batch_size=32):
        """Train the model with differential privacy."""

        # Ensure model is on correct device
        self.model.to(self.device)
        self.model.train()

        # Create optimizer and loss function
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train.float())
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Attach the privacy engine
        self.privacy_engine = PrivacyEngine()
        self.model, optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=max_grad_norm,
        )

        # Train using memory-efficient batches
        with BatchMemoryManager(
            data_loader=self.train_loader,
            max_physical_batch_size=batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:

            for epoch in range(epochs):
                epoch_loss = 0.0

                for batch_X, batch_y in memory_safe_data_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if epoch % 2 == 0:
                    epsilon = self.privacy_engine.get_epsilon(self.delta)
                    print(f"Client {self.client_id} - Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, ε = {epsilon:.2f}")

        return self.privacy_engine.get_epsilon(self.delta)