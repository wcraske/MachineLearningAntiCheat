import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import tqdm
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

#load telemetry data and convert timestamp to datetime format 
data = pd.read_csv('data/telemetryDataFull.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

#convert timestamps to seconds since first timestamp 
data['time_seconds'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds()

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

#add temporal features to the dataset
data = add_temporal_features(data)



#feature selection for model training 
base_features = [
    'time_seconds', 'aim_offset_x', 'aim_offset_y',
    'hit_x', 'hit_y', 'hit_z',
    'delta_aim_x', 'delta_aim_y',
    'delta_hit_x', 'delta_hit_y', 'delta_hit_z',
    'acc_aim_x', 'acc_aim_y'
]
#temporal features to include
temporal_feature_keywords = ['mean_', 'var_20', 'std_', 'interval_']

#collect matching columns
temporal_features = [
    col for col in data.columns
    if any(keyword in col for keyword in temporal_feature_keywords)
]

#final selected feature columns
feature_cols = base_features + temporal_features

#normalize features to zero mean and unit variance for better training stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[feature_cols])

#convert to pytorch tensors
X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(data['aimbot'].values, dtype=torch.float32).reshape(-1, 1)

#class distribution = 58% aimbot data, 42% legit data
#imbalance will be handled later with weighted loss function
print("Class balance:\n", data['aimbot'].value_counts(normalize=True))

#create sequences of fixed length data for LSTM training
#each sequence contains 10 consecutive time steps with their features
def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    
    #sliding window approach to create overlapping sequences
    #each sequence captures temporal behaviour 
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    
    return torch.stack(X_seq), torch.stack(y_seq)

#create sequences with length 10 for temporal pattern recognition
seq_length = 10
X_seq, y_seq = create_sequences(X, y, seq_length)

#temporal grouping for cross validation
#group data by 30 second chunks to prevent data leakage between train/validation
group_ids = (data['time_seconds'] // 30).astype(int) 
group_ids_seq = group_ids[seq_length - 1:].reset_index(drop=True)

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

#handle class imbalance using weighted binary cross entropy loss
#calculate positive class weight to balance the 58%/42% distribution
pos_weight = (y_seq.shape[0] - y_seq.sum()) / y_seq.sum()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.squeeze())

#training function with F1 score tracking and early stopping based on best validation F1
def model_train(model, X_train, y_train, X_val, y_val):
    #adam optimizer with conservative learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    #max epochs with early stopping to prevent overfitting
    n_epochs = 250  
    #small batch size
    batch_size = 10  
    batch_start = torch.arange(0, len(X_train), batch_size)

    #early stopping variables to prevent overfitting
    best_f1 = -np.inf
    best_weights = None

    #training loop 
    for epoch in range(n_epochs):
        model.train()
        
        #mini batch training
        for start in tqdm.tqdm(batch_start, desc=f"Epoch {epoch}", disable=True):
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            #forward pass and loss calculation
            y_logits = model(X_batch)
            loss = loss_fn(y_logits, y_batch)

            #backward pass and parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #validation phase, evaluate model performance without gradient updates
        model.eval()
        with torch.no_grad():
            y_logits = model(X_val)
            #convert logits to binary predictions
            y_pred = torch.sigmoid(y_logits).round()  
            f1 = f1_score(y_val.cpu(), y_pred.cpu())
            
            #save best model weights based on validation F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_weights = copy.deepcopy(model.state_dict())

    #load best weights before returning
    model.load_state_dict(best_weights)
    return best_f1

#split data into train/test sets with stratification to maintain class balance
#70% for training, 30% for final testing
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    X_seq, y_seq, group_ids_seq, train_size=0.7, shuffle=True, stratify=y_seq.numpy())

#group k fold cross validation 
kfold = GroupKFold(n_splits=5)
cv_scores_lstm = []

#5 fold cross validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train.numpy(), groups=group_train)):
    #create fresh model for each fold to ensure independent training
    model = SequenceModel(input_size=X_seq.shape[2])
    
    #train model on current fold and get best validation F1 score
    f1 = model_train(model, X_train[train_idx], y_train[train_idx], X_train[val_idx], y_train[val_idx])
    
    print(f"Fold {fold+1} F1 Score: {f1:.4f}")
    cv_scores_lstm.append(f1)

#calculate final cross validation performance
lstm_f1 = np.mean(cv_scores_lstm)
lstm_std = np.std(cv_scores_lstm)

#display cross validation results
print(f"\nLSTM Cross-Validation F1 Score: {lstm_f1*100:.2f}% (+/- {lstm_std*100:.2f}%)")
print(f"Class distribution:\n{data['aimbot'].value_counts()}")

#train final model on all training data for testing on held out test set
print("FINAL MODEL TRAINING AND TESTING")

#create final model using all training data
final_model = SequenceModel(input_size=X_seq.shape[2])
final_f1 = model_train(final_model, X_train, y_train, X_test, y_test)

#test the final model on the held out test set
final_model.eval()
with torch.no_grad():
    #get predictions on test set
    test_logits = final_model(X_test)
    test_probs = torch.sigmoid(test_logits)
    test_predictions = test_probs.round()
    
    #calculate test metrics
    test_f1 = f1_score(y_test.cpu(), test_predictions.cpu())
    test_accuracy = (test_predictions == y_test).float().mean()
    
    print(f"\nFinal Test Set Performance:")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    #detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test.cpu(), test_predictions.cpu(), 
                              target_names=['Legitimate', 'Aimbot']))
    
    #confusion matrix for detailed analysis
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test.cpu(), test_predictions.cpu())
    print(cm)
    print("(True Negatives: {}, False Positives: {})".format(cm[0,0], cm[0,1]))
    print("(False Negatives: {}, True Positives: {})".format(cm[1,0], cm[1,1]))

#save the trained model and scaler for future use
torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler': scaler,
    'feature_cols': feature_cols,
    'seq_length': seq_length,
    'model_config': {
        'input_size': X_seq.shape[2],
        'hidden_size': int(np.sqrt(X_seq.shape[2]) * 10)
    }
}, 'trained_aimbot_model.pth')

print(f"\nModel saved to 'trained_aimbot_model.pth'")