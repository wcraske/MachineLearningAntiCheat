import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

#LSTM based sequence classifier for aimbot detection
class SequenceModel(nn.Module): 
    def __init__(self, input_size):
        super().__init__()
        #set hidden size based on input dimensionality
        #use square root scaling with multiplier for balanced capacity
        hidden_size = int(np.sqrt(input_size) * 10)
        
        #bidirectional LSTM layer to capture patterns in both time directions
        #single layer to prevent overfitting on small sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        
        #layer normalization on LSTM output for training stability
        #normalizes the concatenated forward and backward hidden states
        self.norm = nn.LayerNorm(hidden_size * 2)   
        
        #fully connected layers for final classification
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        
        #dropout for regularization to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        
        #output layer for binary classification, aimbot vs legit
        self.fc2 = nn.Linear(32, 1)


    #forward pass through the sequence model, gets logits for binary classification
    def forward(self, x):
        #process sequences through bidirectional LSTM
        lstm_out, _ = self.lstm(x)
        
        #take the last timestep output
        #apply layer normalization for stable gradients
        x = self.norm(lstm_out[:, -1, :])
        
        #apply relu and dropout for regularization
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        #output raw logits
        return self.fc2(x)  
    

#get temporal features to identify time based patterns in player behavior
#these features capture behavioral consistency, movement patterns, and firing rhythm
def add_temporal_features(data, window_sizes=[10, 20]):
    #add temporal features using rolling windows to capture player behavior patterns
    for window in window_sizes:
        #rolling variance and mean of aim offsets for stability and movement patterns
        #high variance = crazy aiming, low variance = aimbot stability
        data[f'aim_offset_x_var_{window}'] = data['aim_offset_x'].rolling(window).var()
        data[f'aim_offset_y_var_{window}'] = data['aim_offset_y'].rolling(window).var()
        data[f'aim_offset_x_mean_{window}'] = data['aim_offset_x'].rolling(window).mean()
        data[f'aim_offset_y_mean_{window}'] = data['aim_offset_y'].rolling(window).mean()
        
        #rolling standard deviation of hit positions for consistent accuray that measures how spread out the hit positions are over time windows
        data[f'hit_pos_std_{window}'] = data[['hit_x', 'hit_y', 'hit_z']].rolling(window).std().mean(axis=1)
        
        #rolling mean of shot intervals to capture firing patterns
        #regular intervals may indicate automated firing patterns
        data[f'shot_interval_{window}'] = data['time_seconds'].diff().rolling(window).mean()

    #first and second order differences for aim and hit positions to capture changes in behavior
    data['delta_aim_x'] = data['aim_offset_x'].diff()
    data['delta_aim_y'] = data['aim_offset_y'].diff()
    data['delta_hit_x'] = data['hit_x'].diff()
    data['delta_hit_y'] = data['hit_y'].diff()
    data['delta_hit_z'] = data['hit_z'].diff()
    
    #sudden changes in accleratation = aimbot
    data['acc_aim_x'] = data['delta_aim_x'].diff()
    data['acc_aim_y'] = data['delta_aim_y'].diff()

    return data.fillna(0)

#create sequences of fixed length data for LSTM tetsing
#each sequence contains 10 consecutive time steps with their features
def create_sequences(X, seq_length=10):
    sequences = []
    #sliding window approach to create overlapping sequences
    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i:i+seq_length])

    return torch.stack(sequences)


#load the trained model checkpoint with all configuration data
#contains model weights, feature columns, scaler, and sequence length
checkpoint = torch.load('trained_aimbot_model.pth', map_location=torch.device('cpu'), weights_only=False)

#extract model configuration parameters from checkpoint
#input size determines the number of features the model expects
input_size = checkpoint['model_config']['input_size']

#sequence length defines how many timesteps to analyze together
#must match training sequence length for proper prediction
seq_length = checkpoint['seq_length']

#feature columns list ensures we use the same features as training
#order and selection must be identical to training data
feature_cols = checkpoint['feature_cols']

#scaler object for feature normalization using training statistics
#applies same mean and standard deviation transforms as training
scaler = checkpoint['scaler']

#initialize model architecture with saved config
model = SequenceModel(input_size)

#load the trained weights into model architecture
#puts model in evaluation mode to disable dropout and batch norm updates
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()



#load new telemetry data for aimbot detection analysis
new_data = pd.read_csv('data/testData/telemetryTestDataMix.csv')

#convert timestamp strings to datetime objects for time calculations
new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])

#create relative time in seconds from start for temporal analysis
#enables calculation of shot intervals and movement speeds
new_data['time_seconds'] = (new_data['timestamp'] - new_data['timestamp'].min()).dt.total_seconds()

#add all temporal features using the same function as training
#creates rolling statistics, deltas, and acceleration features
new_data = add_temporal_features(new_data)

#apply exact same normalization as training data using saved statistics
#prevents distribution mismatch between training and inference
X_new_scaled = scaler.transform(new_data[feature_cols])

#convert to pytorch tensor for model input
#float32 matches training precision for consistent results
X_new = torch.tensor(X_new_scaled, dtype=torch.float32)

#generate overlapping sequences of specified length for LSTM processing
#each sequence represents a temporal window of player behavior
X_new_seq = create_sequences(X_new, seq_length)



#disable gradient computation for inference efficiency and memory savings
with torch.no_grad():
    #forward pass through model to get raw logits
    #logits represent unnormalized prediction scores
    logits = model(X_new_seq)
    
    #apply sigmoid to convert logits to probabilities between 0 and 1
    #probability represents confidence that sequence contains aimbot behavior
    probs = torch.sigmoid(logits)
    
    #threshold probabilities at 0.5 to get binary predictions
    #1 = aimbot detected, 0 = legitimate player behavior
    preds = probs.round()


#align predictions with sequence end timestamps
#skip first seq_length-1 rows since they don't have complete sequences
#each prediction corresponds to the last timestamp in its sequence
output_df = new_data.iloc[seq_length - 1:].copy()

#add binary prediction column for easy filtering and analysis
#flattens tensor to 1D array for dataframe compatibility
output_df['aimbot_prediction'] = preds.numpy().flatten()

#add probability scores for confidence assessment and threshold tuning
#higher probabilities indicate stronger aimbot evidence
output_df['aimbot_probability'] = probs.numpy().flatten()


#export results with timestamps and probabilities for further analysis
#includes all original telemetry data plus prediction results
output_df.to_csv('predicted_aimbot_output.csv', index=False)
print("Predictions saved to 'predicted_aimbot_output.csv'")



#display each prediction with timestamp and confidence score
#helps identify specific moments when aimbot behavior was detected
print("\n--- Individual Predictions ---")
for i, row in output_df.iterrows():
    timestamp = row['timestamp']
    prob = row['aimbot_probability']
    pred = int(row['aimbot_prediction'])
    print(f"{timestamp} - Aimbot: {pred} (Prob: {prob:.2f})")

# Print summary statistics
#calculate overall detection rates and distribution of predictions
#provides quick overview of aimbot prevalence in the dataset
total = len(output_df)
aimbot_count = int(output_df['aimbot_prediction'].sum())
not_aimbot_count = total - aimbot_count

print("\n--- Summary Statistics ---")
print(f"Total Sequences Evaluated: {total}")
print(f"Aimbot Predictions: {aimbot_count} ({aimbot_count / total:.2%})")
print(f"Not Aimbot Predictions: {not_aimbot_count} ({not_aimbot_count / total:.2%})")