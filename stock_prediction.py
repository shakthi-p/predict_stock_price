# --- Import required libraries ---
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load CSV file ---
file_path = "C:/Users/BTR/OneDrive/Desktop/Predict_stock_prices_using_LSTM_in_PyTorch/StockData.csv"
df = pd.read_csv(file_path)
print("CSV file loaded successfully.")

# --- Drop unnecessary column ---
if 'Unnamed: 0' in df.columns:
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

# --- Convert Date column ---
df['Date'] = pd.to_datetime(df['Date'])

# --- Feature engineering: add MA7 (7-day moving average) ---
df['MA7'] = df['Close'].rolling(7).mean()
df = df.dropna().reset_index(drop=True)  # Drop rows with NaN values after MA7
print(f"Data shape after adding MA7: {df.shape}")

# --- Select features ---
features = ['High', 'Low', 'Open', 'Close', 'MA7']
price = df[features].values
print(f"Features selected: {features}, shape: {price.shape}")

# --- Split into train and test ---
test_set_size = 20
train_prices = price[:-test_set_size]
test_prices = price[-test_set_size:]
print(f"Train shape: {train_prices.shape}, Test shape: {test_prices.shape}")

# --- Scale features ---
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_prices)
train_scaled = scaler.transform(train_prices)
test_scaled = scaler.transform(test_prices)
print("Features scaled successfully.")

# --- Create sequences ---
train_window = 14

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 3]  # predict Close
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_scaled, train_window)
X_test, y_test = create_sequences(np.concatenate((train_scaled[-train_window:], test_scaled)), train_window)
print(f"Sequences created. X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

# --- Split train into train + validation ---
val_size = 0.1
val_index = int(len(X_train)*(1-val_size))
X_val, y_val = X_train[val_index:], y_train[val_index:]
X_train, y_train = X_train[:val_index], y_train[:val_index]
print(f"After split -> X_train={X_train.shape}, X_val={X_val.shape}")

# --- Convert to PyTorch tensors and DataLoader ---
batch_size = 16
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- LSTM Model ---
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=1, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers,
                            dropout=dropout, batch_first=False)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # input_seq: (seq_len, batch, features)
        h0 = torch.zeros(2, input_seq.size(1), self.hidden_layer_size).to(device)
        c0 = torch.zeros(2, input_seq.size(1), self.hidden_layer_size).to(device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[-1])
        return predictions

input_size = X_train.shape[2]
model = LSTM(input_size=input_size).to(device)
print("LSTM model initialized.")

# --- Loss function, optimizer, scheduler ---
loss_function = nn.SmoothL1Loss()  # Huber loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# --- Training with early stopping ---
epochs = 20
patience = 5
best_val_loss = np.inf
early_stop_counter = 0
epoch_losses = []

print("Training started...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for seq, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True):
        seq = seq.permute(1,0,2).to(device)  # seq_len, batch, features
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq)
        y_pred = y_pred.view(-1)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seq, labels in val_loader:
            seq = seq.permute(1,0,2).to(device)
            labels = labels.to(device)
            y_pred = model(seq)
            y_pred = y_pred.view(-1)
            val_loss += loss_function(y_pred, labels).item()
    avg_val_loss = val_loss / len(val_loader)
    
    epoch_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Scheduler step
    scheduler.step()
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        # Save best model
        torch.save(model.state_dict(), "best_lstm_model.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training completed.")

# --- Plot training loss ---
plt.plot(range(len(epoch_losses)), epoch_losses)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.show()

# --- Load best model ---
model.load_state_dict(torch.load("best_lstm_model.pth"))
model.eval()

# --- Evaluation ---
predictions = []
with torch.no_grad():
    for i in range(X_test.shape[0]):
        seq = torch.FloatTensor(X_test[i]).unsqueeze(1).to(device)
        pred = model(seq)
        predictions.append(pred.item())
print("Evaluation completed.")

# --- Inverse transform Close ---
close_scaler = MinMaxScaler(feature_range=(-1,1))
close_scaler.fit(train_prices[:,3].reshape(-1,1))
pred_new = close_scaler.inverse_transform(np.array(predictions).reshape(-1,1))
actual_new = close_scaler.inverse_transform(y_test.reshape(-1,1))

# --- Metrics ---
mae = mean_absolute_error(actual_new, pred_new)
rmse = np.sqrt(mean_squared_error(actual_new, pred_new))
mape = np.mean(np.abs((actual_new - pred_new)/actual_new)) * 100
print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")

# --- Plot predictions vs actual ---
plt.figure(figsize=(10,5))
plt.plot(actual_new, 'r-', label='Actual')
plt.plot(pred_new, 'c-', label='Predicted')
plt.ylabel('Stock Value (dollars)')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()

# --- Plot prediction error ---
difference = actual_new - pred_new
plt.figure(figsize=(10,3))
plt.plot(difference, 'b')
plt.ylabel('Difference (Actual - Predicted)')
plt.title('Prediction Error')
plt.show()

# --- Optional: Error distribution ---
plt.figure(figsize=(8,4))
plt.hist(difference, bins=20, color='orange')
plt.title("Prediction Error Distribution")
plt.show()
