import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from utils.database import get_database
from core.risk_manager import RiskManager
from core.executor import OrderExecutor
import json

app = FastAPI(title="Elite Trading Dashboard")
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, os.getenv("DASHBOARD_USERNAME", "admin"))
    correct_password = secrets.compare_digest(credentials.password, os.getenv("DASHBOARD_PASSWORD", "admin"))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/status")
def get_status(username: str = Depends(authenticate)):
    risk_manager = RiskManager()
    executor = OrderExecutor()
    status_obj = risk_manager.get_status()
    balance = executor.get_balance()
    return {
        "balance": balance,
        "risk_level": status_obj.level.value,
        "current_drawdown": status_obj.current_drawdown,
        "daily_pnl": status_obj.daily_pnl,
        "open_positions": status_obj.open_positions,
        "phase": status_obj.phase.value
    }

@app.get("/priority")
def get_priority(username: str = Depends(authenticate)):
    try:
        with open("data/priority_list.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "priority_list.json not found"}

@app.get("/logs")
def get_logs(username: str = Depends(authenticate)):
    try:
        with open("data/logs/trades.log", "r") as f:
            lines = f.readlines()
            # Return last 50 lines
            return {"logs": lines[-50:]}
    except FileNotFoundError:
        return {"error": "trades.log not found"}

@app.get("/trades")
def get_trades(username: str = Depends(authenticate)):
    db = get_database()
    return {"trades": db.get_recent_trades(limit=50)}

def start_dashboard(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)
