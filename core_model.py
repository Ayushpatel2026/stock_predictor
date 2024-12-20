from matplotlib import pyplot as plt
import data_analysis as da
import mlp as mlp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle

# Set seeds for reproducibility - only for testing purposes (TURN OFF WHEN TRAINING FINAL MODEL)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Make PyTorch deterministic - this will slow down the code but will make the results reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(42)

data = da.load_and_clean_data('data/all_stocks_5yr.csv')
training = ['high', 'open', 'low', 'volume', 'year', 'month', 'day', 'name_encoded']

# Split the data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = da.split_data(data, training)

normal_scaler_x = MinMaxScaler()
X_train_scaled = normal_scaler_x.fit_transform(X_train)
X_test_scaled = normal_scaler_x.transform(X_test)
X_val_scaled = normal_scaler_x.transform(X_val)

normal_scaler_y = MinMaxScaler()
Y_train_scaled = normal_scaler_y.fit_transform(y_train.values.reshape(-1, 1))
Y_test_scaled = normal_scaler_y.transform(y_test.values.reshape(-1, 1))
Y_val_scaled = normal_scaler_y.transform(y_val.values.reshape(-1, 1))

# store the scalers for later use
with open('data/scaler_x.pkl', 'wb') as f:
    pickle.dump(normal_scaler_x, f)
with open('data/scaler_y.pkl', 'wb') as f:
    pickle.dump(normal_scaler_y, f)

# Convert data to PyTorch tensors
inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
targets = torch.tensor(Y_train_scaled, dtype=torch.float32)

input_size = len(training)  # 8 features
model = mlp.MLP(input_size, hidden_size1=128, hidden_size2=64, hidden_size3=64)

# Print model architecture
print(model)

#==============================================================================
# Train the Model

# for reproducibility
g = torch.Generator().manual_seed(42)

num_epochs = 10000
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
batch_size = 256

# Create a TensorDataset and DataLoader for minibatch training
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, generator=g, shuffle=True)
train_losses = []
epoch_losses = []
val_losses = []

# Learning rate decay - decrease LR every 2000 epochs by multiplying by 0.8
scheduler = StepLR(optimizer, step_size=2000, gamma=0.8)

# training loop
for epoch in range(num_epochs):
    # Get a single batch from the dataloader (this will be random each epoch)
    batch_inputs, batch_targets = next(iter(dataloader))
    # Forward pass
    outputs = model(batch_inputs)
    loss = criterion(outputs, batch_targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()

    # Update the parameters
    optimizer.step()

    # update the learning rate
    scheduler.step()

    train_losses.append(loss.item())
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')
        epoch_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val_scaled, dtype=torch.float32)
            val_targets = torch.tensor(Y_val_scaled, dtype=torch.float32)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_losses.append(val_loss.item())
            print(f'Validation Loss: {val_loss.item():.7f}')
            model.train()

# Plot the training and val_losses vs the number of epochs
# make the val loss dots bigger
plt.figure(figsize=(10, 6))
train_losses = train_losses[1000:] # remove the first 1000 epochs for better visualization
plt.plot(train_losses, label='Train Loss')
plt.scatter(range(0, num_epochs, 1000), val_losses, label='Validation Loss', color='orange', s=40)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# ==============================================================================
# Test the Model and Make Predictions

def test_model(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_targets = torch.tensor(Y_test, dtype=torch.float32)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        print(f'Test Loss: {test_loss.item():.7f}')

test_model(model, X_test_scaled, Y_test_scaled)

# Make predictions

def evaluate_model(model, X_test, Y_test, normal_scaler_y):
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_outputs = model(test_inputs)

        # Inverse transform the scaled predictions
        test_predictions = normal_scaler_y.inverse_transform(test_outputs.cpu().numpy())
        test_actuals = normal_scaler_y.inverse_transform(Y_test)

        # Calculate MAPE - average percentage difference between predictions and actual values
        mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100
        print(f'Mean Absolute Percentage Error: {mape:.2f}%')

        # calculate RMSE - gives rough estimation of how far the model's predictions are from the actual values
        rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
        print(f'Root Mean Squared Error: {rmse:.2f}')
    
evaluate_model(model, X_test_scaled, Y_test_scaled, normal_scaler_y)

# Save the model
torch.save(model.state_dict(), 'models/core_mlp_model.pth')