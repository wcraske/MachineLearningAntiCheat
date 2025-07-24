import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# === 1. Model Class ===
class SequenceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = int(np.sqrt(input_size) * 10)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.norm(lstm_out[:, -1, :])
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# === 2. Temporal Feature Engineering ===
def add_temporal_features(data, window_sizes=[10, 20]):
    for window in window_sizes:
        data[f'aim_offset_x_var_{window}'] = data['aim_offset_x'].rolling(window).var()
        data[f'aim_offset_y_var_{window}'] = data['aim_offset_y'].rolling(window).var()
        data[f'aim_offset_x_mean_{window}'] = data['aim_offset_x'].rolling(window).mean()
        data[f'aim_offset_y_mean_{window}'] = data['aim_offset_y'].rolling(window).mean()
        data[f'hit_pos_std_{window}'] = data[['hit_x', 'hit_y', 'hit_z']].rolling(window).std().mean(axis=1)
        data[f'shot_interval_{window}'] = data['time_seconds'].diff().rolling(window).mean()
    data['delta_aim_x'] = data['aim_offset_x'].diff()
    data['delta_aim_y'] = data['aim_offset_y'].diff()
    data['delta_hit_x'] = data['hit_x'].diff()
    data['delta_hit_y'] = data['hit_y'].diff()
    data['delta_hit_z'] = data['hit_z'].diff()
    data['acc_aim_x'] = data['delta_aim_x'].diff()
    data['acc_aim_y'] = data['delta_aim_y'].diff()
    return data.fillna(0)

# === 3. Create Sequences ===
def create_sequences(X, seq_length=10):
    X_seq = [X[i:i+seq_length] for i in range(len(X) - seq_length + 1)]
    return torch.stack(X_seq)

# === 4. Load Model & Metadata ===
checkpoint = torch.load('trained_aimbot_model.pth', map_location=torch.device('cpu'), weights_only=False)
input_size = checkpoint['model_config']['input_size']
seq_length = checkpoint['seq_length']
feature_cols = checkpoint['feature_cols']
scaler = checkpoint['scaler']

model = SequenceModel(input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === 5. Load and Process New Data ===
new_data = pd.read_csv('data/testData/telemetryTestDataMix.csv')
new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
new_data['time_seconds'] = (new_data['timestamp'] - new_data['timestamp'].min()).dt.total_seconds()

new_data = add_temporal_features(new_data)

# Normalize with saved scaler
X_new_scaled = scaler.transform(new_data[feature_cols])
X_new = torch.tensor(X_new_scaled, dtype=torch.float32)

# Create sequences
X_new_seq = create_sequences(X_new, seq_length)

# === 6. Predict ===
with torch.no_grad():
    logits = model(X_new_seq)
    probs = torch.sigmoid(logits)
    preds = probs.round()

# === 7. Save Predictions ===
# Align predictions with sequence end timestamps
output_df = new_data.iloc[seq_length - 1:].copy()
output_df['aimbot_prediction'] = preds.numpy().flatten()
output_df['aimbot_probability'] = probs.numpy().flatten()

# Save to CSV
output_df.to_csv('predicted_aimbot_output.csv', index=False)
print("Predictions saved to 'predicted_aimbot_output.csv'")

# === 8. Print Predictions & Stats ===
# Print individual predictions
print("\n--- Individual Predictions ---")
for i, row in output_df.iterrows():
    timestamp = row['timestamp']
    prob = row['aimbot_probability']
    pred = int(row['aimbot_prediction'])
    print(f"{timestamp} - Aimbot: {pred} (Prob: {prob:.2f})")

# Print summary statistics
total = len(output_df)
aimbot_count = int(output_df['aimbot_prediction'].sum())
not_aimbot_count = total - aimbot_count

print("\n--- Summary Statistics ---")
print(f"Total Sequences Evaluated: {total}")
print(f"Aimbot Predictions: {aimbot_count} ({aimbot_count / total:.2%})")
print(f"Not Aimbot Predictions: {not_aimbot_count} ({not_aimbot_count / total:.2%})")
