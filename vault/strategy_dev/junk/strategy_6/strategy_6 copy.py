import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os

# Define constants
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOOKBACK_WINDOW = 20  # Days to consider for historical input
TARGET_CLASSES = [-2, -1, 0, 1, 2]  # Market regimes
MODEL_SAVE_PATH = "model_checkpoints"  # Directory to save model checkpoints

# Load OHLCV data from disk
def load_data(file_path):
    all_files = glob.glob(file_path + "/*.csv")
    data_list = []
    for file in all_files[:10]:  # Limit to first 10 files for demonstration
        df = pd.read_csv(file, parse_dates=['datetime'])
        required_columns = ['datetime','close', 'high', 'low', 'open', 'volume', 'symbol']
        available_columns = [col for col in required_columns if col in df.columns]
        missing_columns = set(required_columns) - set(available_columns)

        # Select available columns and fill missing data
        df = df[available_columns]
        df = df.dropna()  # Drop rows with missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Ensure numeric type
        df = df.dropna()  # Drop rows with invalid numeric data
        data_list.append(df)

    return pd.concat(data_list, ignore_index=True)

# Prepare dataset class
class MarketDataset(Dataset):
    def __init__(self, data, symbols):
        self.data = data
        self.symbols = symbols

    def __len__(self):
        return len(self.data) - LOOKBACK_WINDOW

    def __getitem__(self, idx):
        x = self.data[idx:idx + LOOKBACK_WINDOW]
        y = self._get_label(idx + LOOKBACK_WINDOW)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _get_label(self, idx):
        future_return = (self.data[idx, 2] - self.data[idx - 1, 2]) / self.data[idx - 1, 2]  # Using close price
        if future_return > 0.03:
            return 4  # +2 regime mapped to 4
        elif future_return > 0.01:
            return 3  # +1 regime mapped to 3
        elif future_return > -0.01:
            return 2  # Neutral mapped to 2
        elif future_return > -0.03:
            return 1  # -1 regime mapped to 1
        else:
            return 0  # -2 regime mapped to 0

# Generative AI-inspired model
class WorldModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WorldModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        dec_out, _ = self.decoder(hidden.permute(1, 0, 2))
        pred = self.classifier(dec_out[:, -1, :])
        return pred

# Train-test split and data preparation
def prepare_data(data, test_size=0.2):
    symbols = data['symbol'].unique()
    data_by_symbol = [data[data['symbol'] == sym].values[:, 1:-1] for sym in symbols]
    combined_data = np.concatenate(data_by_symbol, axis=0)
    train_data, test_data = train_test_split(combined_data, test_size=test_size, shuffle=False)
    return train_data, test_data, symbols

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs, start_epoch=0):
    try:
      model.train()
      os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
      for epoch in range(start_epoch, epochs):
          total_loss = 0
          for batch in dataloader:
              inputs, labels = batch
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
              total_loss += loss.item()

          print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

          # Save model every 5 epochs
          if (epoch + 1) % 5 == 0:
              checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch + 1}.pth")
              torch.save(model.state_dict(), checkpoint_path)
              print(f"Model saved at {checkpoint_path}")
    except KeyboardInterrupt:
      print('Training interrupted...')
      # Saving the model before exiting
      checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch + 1}.pth")
      torch.save(model.state_dict(), checkpoint_path)
      print(f"Model saved at {checkpoint_path}")
# Backtesting function
def backtest(model, test_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Backtesting Accuracy: {accuracy:.4f}")
    return accuracy

# Main execution
if __name__ == "__main__":
    file_path = "/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/data/ibkr/1d"  # Replace with your data directory
    raw_data = load_data(file_path)
    train_data, test_data, symbols = prepare_data(raw_data)

    train_dataset = MarketDataset(train_data, symbols)
    test_dataset = MarketDataset(test_data, symbols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_data.shape[1]  # Adjusted for OHLCV
    hidden_dim = 128
    output_dim = len(TARGET_CLASSES)

    model = WorldModel(input_dim, hidden_dim, output_dim)

    # Load the latest model checkpoint if available
    if os.path.exists(MODEL_SAVE_PATH):
        checkpoints = sorted(glob.glob(os.path.join(MODEL_SAVE_PATH, "model_epoch_*.pth")))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            model.load_state_dict(torch.load(latest_checkpoint))
            print(f"Loaded model from {latest_checkpoint}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    print("Backtesting the model...")
    backtest(model, test_loader)