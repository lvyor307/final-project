import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import pickle

from Model import AudioLSTM


class HyperParameterTuning:
    def __init__(self, param_grid: dict):
        self.param_grid = param_grid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a list of dictionaries for each combination
        self.list_of_hp_dicts = list(ParameterGrid(param_grid))
        self.n_model = 1
        self.accuracy_dict = {}

    def _train(self, X_train: DataLoader, X_devel: DataLoader,
              input_size: int, hidden_size: int,
              num_layers: int, num_classes: int,
              criterion: any, optimizer: any, num_epochs: int, learning_rate: float):
        """
        Train the model
        :return:
        """
        # Define the model architecture (AudioLSTM in this case)
        model = AudioLSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          num_classes=num_classes)
        # Move model to GPU if available
        model.to(self.device)
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        # Training loop
        for epoch in range(num_epochs):
            # Set model to training mode
            model.train()

            # Iterate over the training dataset
            for inputs, labels in X_train:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                # Compute loss
                loss = criterion(outputs, labels)
                # Backward pass
                loss.backward()
                # Update self.model parameters
                optimizer.step()

            # Validate the self.model on the development set
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                for inputs, labels in X_devel:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                accuracy = total_correct / total_samples
                print(f"{self.n_model} === Epoch [{epoch + 1}/{num_epochs}], devel Accuracy: {accuracy:.4f}")
        model_name = f"model_{self.n_model}_low"
        with open(f'{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        self.n_model += 1
        return accuracy

    def fit(self, X_train: DataLoader, X_devel: DataLoader):
        """

        :param X_train:
        :param X_devel:
        :return:
        """
        for hp_set in self.list_of_hp_dicts:
            accuracy = self._train(X_train, X_devel, **hp_set)
            self.accuracy_dict.update({str(self.n_model): accuracy})

    def print_best_model(self):
        """
        Get the best hyperparameters
        :return:
        """
        # Get the key-value pair with the maximum value
        max_pair = max(self.accuracy_dict.items(), key=lambda x: x[1])
        print(max_pair)
# model 15 acc = 0.45213849287169044