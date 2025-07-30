#uvicorn telemetryLiveAPI:app --reload

#to view docs
#http://127.0.0.1:8000/docs 

#to download csv
#http://127.0.0.1:8000/telemetry/csv

#to view live predictions
#http://127.0.0.1:8000/static/index.html

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import sqlite3
from datetime import datetime, timedelta
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import io
import csv
import threading
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler



#create fastapi app and sqlite3 connection
app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
conn = sqlite3.connect("aim_telemetry.db", check_same_thread=False)
c = conn.cursor()


#schema for databases
c.execute('''
CREATE TABLE IF NOT EXISTS aim_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    player_id TEXT,
    aim_offset_x REAL,
    aim_offset_y REAL,
    hit_x REAL,
    hit_y REAL,
    hit_z REAL
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    player_id TEXT,
    aimbot_prediction INTEGER,
    aimbot_probability REAL,
    prediction_timestamp TEXT
)
''')
conn.commit()
#base model for telemetry data for api
class Telemetry(BaseModel):
    player_id: str
    aim_offset_x: float
    aim_offset_y: float
    hit_x: float
    hit_y: float
    hit_z: float

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



#load model and metadata
try:
    checkpoint = torch.load('trained_aimbot_model.pth', map_location=torch.device('cpu'), weights_only=False)
    input_size = checkpoint['model_config']['input_size']
    seq_length = checkpoint['seq_length']
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['scaler']
    
    model = SequenceModel(input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#global variables for predictions
latest_predictions = []
prediction_lock = threading.Lock()



#prediction function to run on recent telemetry data
#
def run_predictions():
    if model is None:
        print("Model not loaded, skipping predictions")
        return
    
    try:
        #get data from last 5 minutes to ensure we have enough for sequences
        five_min_ago = (datetime.now() - timedelta(minutes=5)).isoformat()
        c.execute("""SELECT * FROM aim_logs 
                     WHERE timestamp > ? 
                     ORDER BY timestamp ASC""", (five_min_ago,))
        rows = c.fetchall()
        
        if len(rows) < seq_length:
            print(f"Not enough data for prediction (need {seq_length}, got {len(rows)})")
            return
        
        #convert to df
        df = pd.DataFrame(rows, columns=[
            'id', 'timestamp', 'player_id', 'aim_offset_x', 
            'aim_offset_y', 'hit_x', 'hit_y', 'hit_z'
        ])
        
        #process by player
        global latest_predictions
        current_predictions = []
        
        for player_id in df['player_id'].unique():
            player_data = df[df['player_id'] == player_id].copy()
            
            if len(player_data) < seq_length:
                continue
                
            #convert timestamp and add time_seconds
            player_data['timestamp'] = pd.to_datetime(player_data['timestamp'])
            player_data['time_seconds'] = (player_data['timestamp'] - player_data['timestamp'].min()).dt.total_seconds()
            
            #add temporal features
            player_data = add_temporal_features(player_data)
            
            #scale features
            X_scaled = scaler.transform(player_data[feature_cols])
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            #create sequences
            X_seq = create_sequences(X_tensor, seq_length)
            if X_seq is None:
                continue
            
            #predict
            with torch.no_grad():
                logits = model(X_seq)
                probs = torch.sigmoid(logits)
                preds = probs.round()
            
            #get the most recent prediction
            if len(preds) > 0:
                latest_prob = float(probs[-1].item())
                latest_pred = int(preds[-1].item())
                latest_timestamp = player_data.iloc[-1]['timestamp']
                
                prediction_entry = {
                    'player_id': player_id,
                    'timestamp': latest_timestamp.isoformat(),
                    'aimbot_prediction': latest_pred,
                    'aimbot_probability': latest_prob,
                    'prediction_timestamp': datetime.now().isoformat()
                }
                
                current_predictions.append(prediction_entry)
                
                #store in database
                c.execute('''INSERT INTO predictions 
                             (timestamp, player_id, aimbot_prediction, aimbot_probability, prediction_timestamp)
                             VALUES (?, ?, ?, ?, ?)''',
                          (latest_timestamp.isoformat(), player_id, latest_pred, latest_prob, 
                           datetime.now().isoformat()))
        
        conn.commit()
        
        #update global predictions
        with prediction_lock:
            latest_predictions = current_predictions
        
        print(f"Predictions updated: {len(current_predictions)} players processed")
        
    except Exception as e:
        print(f"Error in run_predictions: {e}")

#schedule predictions to run every minute
def prediction_scheduler():
    while True:
        run_predictions()
        time.sleep(60)

#start background thread
if model is not None:
    prediction_thread = threading.Thread(target=prediction_scheduler, daemon=True)
    prediction_thread.start()
    print("prediction scheduler started")

#endpoints
@app.post("/telemetry")
def receive_data(data: Telemetry):
    timestamp = datetime.now().isoformat()
    c.execute('''INSERT INTO aim_logs (timestamp, player_id, aim_offset_x, aim_offset_y, hit_x, hit_y, hit_z)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (timestamp, data.player_id, data.aim_offset_x, data.aim_offset_y,
               data.hit_x, data.hit_y, data.hit_z))
    conn.commit()

    print(f"[{timestamp}] {data.player_id} → Aim({data.aim_offset_x:.2f}, {data.aim_offset_y:.2f}) @ ({data.hit_x:.1f}, {data.hit_y:.1f}, {data.hit_z:.1f})")
    return {"status": "received"}

@app.get("/telemetry", response_model=List[Dict])
def get_telemetry():
    """Get all telemetry logs."""
    c.execute("SELECT * FROM aim_logs ORDER BY id DESC LIMIT 1000")
    rows = c.fetchall()

    result = []
    for row in rows:
        telemetry_entry = {
            "id": row[0],
            "timestamp": row[1],
            "player_id": row[2],
            "aim_offset_x": row[3],
            "aim_offset_y": row[4],
            "hit_x": row[5],
            "hit_y": row[6],
            "hit_z": row[7],
        }
        result.append(telemetry_entry)
    return result

@app.get("/predictions")
def get_latest_predictions():
    """Get the latest predictions for all players."""
    with prediction_lock:
        return latest_predictions.copy()

@app.get("/predictions/history")
def get_prediction_history():
    """Get prediction history from database."""
    c.execute("""SELECT * FROM predictions 
                 ORDER BY prediction_timestamp DESC 
                 LIMIT 100""")
    rows = c.fetchall()
    
    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "timestamp": row[1],
            "player_id": row[2],
            "aimbot_prediction": row[3],
            "aimbot_probability": row[4],
            "prediction_timestamp": row[5]
        })
    return result

@app.get("/telemetry/csv")
def download_csv():
    """Returns telemetry data as a CSV file."""
    c.execute("SELECT * FROM aim_logs ORDER BY id ASC")
    rows = c.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "player_id", "aim_offset_x", "aim_offset_y", "hit_x", "hit_y", "hit_z"])
    
    for row in rows:
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=telemetry.csv"
    })

@app.get("/predictions/csv")
def download_predictions_csv():
    """Returns predictions as a CSV file."""
    c.execute("SELECT * FROM predictions ORDER BY prediction_timestamp DESC")
    rows = c.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "player_id", "aimbot_prediction", "aimbot_probability", "prediction_timestamp"])
    
    for row in rows:
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=predictions.csv"
    })