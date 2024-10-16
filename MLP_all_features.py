import torch
import torch.nn as nn

class MLPAllFeatures(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPAllFeatures, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Predicting closing price

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Input size based on all features
input_size = len(['open', 'high', 'low', 'volume', 'year', 'month', 'day', 'name_encoded'])  # Example
hidden_size = 64  # Hyperparameter
model = MLPAllFeatures(input_size, hidden_size)

# Print model architecture
print(model)
