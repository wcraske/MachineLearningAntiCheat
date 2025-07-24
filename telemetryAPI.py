#uvicorn telemetryAPI:app --reload


#to view docs
#http://127.0.0.1:8000/docs 

#to download csv
#http://127.0.0.1:8000/telemetry/csv

#to view html
#http://127.0.0.1:8000/static/index.html


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import sqlite3
from datetime import datetime
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import io
import csv

#create fastapi app and sqlite3 connection
app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
conn = sqlite3.connect("aim_telemetry.db", check_same_thread=False)
c = conn.cursor()

#schema for database
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
conn.commit()

#base model for telemetry data for api
class Telemetry(BaseModel):
    player_id: str
    aim_offset_x: float
    aim_offset_y: float
    hit_x: float
    hit_y: float
    hit_z: float

#aim offset x is yaw , aim offset y is pitch
#hit_x, hit_y, hit_z are the coordinates of the hit in the game world

#endpoint to receive telemetry data
@app.post("/telemetry")
def receive_data(data: Telemetry):
    #gets time
    timestamp = datetime.now().isoformat()
    c.execute('''INSERT INTO aim_logs (timestamp, player_id, aim_offset_x, aim_offset_y, hit_x, hit_y, hit_z)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (timestamp, data.player_id, data.aim_offset_x, data.aim_offset_y,
               data.hit_x, data.hit_y, data.hit_z))
    conn.commit()

    #prints to console for testing
    print(f"[{timestamp}] {data.player_id} → Aim({data.aim_offset_x:.2f}, {data.aim_offset_y:.2f}) @ ({data.hit_x:.1f}, {data.hit_y:.1f}, {data.hit_z:.1f})")

    return {"status": "received"}


#endpoint to get telemetry data
@app.get("/telemetry", response_model=List[Dict])
def get_telemetry():
    """get the all logs."""
    c.execute("SELECT * FROM aim_logs ORDER BY id ASC")
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



@app.get("/telemetry/csv")
def download_csv():
    """Returns telemetry data as a CSV file."""
    c.execute("SELECT * FROM aim_logs ORDER BY id ASC")
    rows = c.fetchall()

    #create in-memory file-like object
    output = io.StringIO()
    writer = csv.writer(output)

    #write header
    writer.writerow(["id", "timestamp", "player_id", "aim_offset_x", "aim_offset_y", "hit_x", "hit_y", "hit_z"])

    #write data rows
    for row in rows:
        writer.writerow(row)

    output.seek(0)

    #return as a streamed CSV file
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=telemetry.csv"
    })