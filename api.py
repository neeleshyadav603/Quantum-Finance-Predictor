# backend/api.py (UPDATED FOR PHASE 2 - DATABASE LOGGING)

import sys
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sqlite3 # <- ADD THIS IMPORT

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

sys.path.append('.')
from models.data_pipeline import create_dataset, get_sentiment
from models.classical_model import train_lstm_model
from models.quantum_model import train_and_predict_quantum_regressor, train_and_predict_quantum_classifier

# --- Path Definitions (No changes) ---
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_PATH = PROJECT_ROOT / "static"
TEMPLATES_PATH = PROJECT_ROOT / "templates"
DB_PATH = PROJECT_ROOT / "data" / "prediction_history.db"

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory=TEMPLATES_PATH)

# --- Pydantic Data Models (No changes) ---
class ConfidenceFactors(BaseModel):
    historical_accuracy: str
    market_sentiment: str
    stock_volatility: str

class AnalysisMetrics(BaseModel):
    ticker: str
    company_name: str
    classical_mse: float
    quantum_class_accuracy: float
    quantum_regr_mse: float
    sentiment_score: float
    # ... (rest of the model is the same)
    risk_level: str
    volatility: str
    market_trend: str
    confidence_score: int
    confidence_factors: ConfidenceFactors
    forecast: Dict[str, float]
    chart_labels: List[str]
    historical_prices: List[float]
    predicted_prices: List[Optional[float]]
    quantum_predictions: List[int]
    quantum_regr_predictions: List[float]


# --- Helper Functions ---
# --- ADD THIS NEW HELPER FUNCTION TO LOG DATA ---
def log_prediction_to_db(ticker: str, classical_mse: float, quantum_accuracy: float):
    """Saves the key results of an analysis run to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        # The SQL query matches your database schema [cite: 40, 41]
        cursor.execute("""
            INSERT INTO predictions (ticker, classical_mse, quantum_accuracy, timestamp)
            VALUES (?, ?, ?, ?)
        """, (ticker, classical_mse, quantum_accuracy, timestamp))
        conn.commit()
        conn.close()
        print(f"Successfully logged prediction for {ticker} to the database.")
    except Exception as e:
        print(f"Database logging failed: {e}")

def calculate_confidence(mse, sentiment, volatility_level):
    # ... (no changes here)
    score = 75
    if mse < 0.005: score += 10
    if abs(sentiment) > 0.5: score += 10
    if volatility_level == "Low": score += 5
    elif volatility_level == "High": score -= 10
    return min(max(score, 50), 98)

def get_risk_assessment(ticker_data: pd.DataFrame):
    # ... (no changes here)
    daily_return = ticker_data['Close'].pct_change().dropna()
    volatility_value = daily_return.std() * 100
    if volatility_value < 1.5: return "Low", f"{volatility_value:.2f}% (Low)"
    elif volatility_value < 3.0: return "Medium", f"{volatility_value:.2f}% (Medium)"
    else: return "High", f"{volatility_value:.2f}% (High)"

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_app(request: Request):
    return templates.TemplateResponse("main_app.html", {"request": request})

@app.get("/analysis/dashboard/{ticker}", response_model=AnalysisMetrics)
async def get_dashboard_metrics(ticker: str):
    try:
        # NEW LINE
        raw_dataframe = await run_in_threadpool(create_dataset, ticker)
        if raw_dataframe is None:
            raise ValueError("Could not fetch data.")
        
        classical_mse, classical_predictions = await run_in_threadpool(train_lstm_model, raw_dataframe)
        quantum_class_accuracy, quantum_class_predictions = await run_in_threadpool(train_and_predict_quantum_classifier, raw_dataframe)
        quantum_regr_mse, quantum_regr_predictions = await run_in_threadpool(train_and_predict_quantum_regressor, raw_dataframe)
        
        sentiment_score = await run_in_threadpool(get_sentiment, ticker)
        risk_level, volatility = get_risk_assessment(raw_dataframe)
        confidence_score = calculate_confidence(classical_mse, sentiment_score, risk_level)
        
        # --- ADD THIS LINE TO CALL THE LOGGING FUNCTION ---
        # We log the results right after they are calculated
        log_prediction_to_db(ticker, classical_mse, quantum_class_accuracy)

        # --- The rest of the function is for formatting and returning the response (no changes) ---
        chart_data = raw_dataframe.tail(60).copy()
        chart_data.index = chart_data.index.strftime('%d %b')
        historical_labels = chart_data.index.tolist()
        # ... (rest of the function is the same)
        historical_prices = [round(price, 2) for price in chart_data['Close'].tolist()]
        prediction_labels = [f"Day +{i+1}" for i in range(30)]
        
        predicted_prices_padded = [None] * len(historical_prices)
        predicted_prices_padded.extend([round(price, 2) for price in classical_predictions])
        
        full_chart_labels = historical_labels + prediction_labels
        
        company_names = {"GOOGL": "Alphabet Inc.", "MSFT": "Microsoft Corp.", "AAPL": "Apple Inc."}

        return AnalysisMetrics(
            ticker=ticker,
            company_name=company_names.get(ticker.upper(), ticker),
            classical_mse=classical_mse,
            quantum_class_accuracy=quantum_class_accuracy,
            quantum_regr_mse=quantum_regr_mse,
            sentiment_score=sentiment_score,
            risk_level=risk_level,
            volatility=volatility,
            market_trend="Neutral",
            confidence_score=confidence_score,
            confidence_factors=ConfidenceFactors(
                historical_accuracy="High" if classical_mse < 0.005 else "Standard",
                market_sentiment="Strong" if abs(sentiment_score) > 0.5 else "Neutral",
                stock_volatility=risk_level
            ),
            forecast={ "Q4 '25": random.uniform(1.5, 2.5), "Q1 '26": random.uniform(2.5, 3.5), "Q2 '26": random.uniform(3.5, 4.5) },
            chart_labels=full_chart_labels,
            historical_prices=historical_prices,
            predicted_prices=predicted_prices_padded,
            quantum_predictions=quantum_class_predictions,
            quantum_regr_predictions=quantum_regr_predictions
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")