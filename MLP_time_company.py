import torch
import torch.nn as nn

class MLPTimeCompany(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, dropout_prob=0.2):
        super(MLPTimeCompany, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)  # Output is the closing price

        # we can experiment with different activation functions and non-linearities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu(out)
        #out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        #out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc3(out)
        #out = self.tanh(out)
        out = self.fc4(out)
        return out
    

