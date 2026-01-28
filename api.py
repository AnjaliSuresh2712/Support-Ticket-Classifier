from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import numpy as np

app = FastAPI(
    title="Support Ticket Classifier API",
    description="REST API for automated support ticket classification",
    version="1.0.0"
)

# Load model and vectorizer
try:
    model = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    print("Models not found, API to run in demo mode")

class TicketRequest(BaseModel):
    text: str = Field(..., description="Support ticket text to classify")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I can't log into my account"
            }
        }

class PredictionResponse(BaseModel):
    category: str = Field(..., description="Predicted ticket category")
    confidence: float = Field(..., description="Prediction confidence score")
    all_probabilities: dict = Field(..., description="Probability for each category")
    
@app.get("/")
def root():
    """API health check endpoint"""
    return {
        "message": "Support Ticket Classifier API",
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(ticket: TicketRequest):
    """
    Classify a support ticket
    
    Returns the predicted category and confidence score
    """
    if not MODEL_LOADED:
        # mock prediction
        return PredictionResponse(
            category="technical",
            confidence=0.85,
            all_probabilities={
                "technical": 0.85,
                "billing": 0.10,
                "account": 0.03,
                "feature_request": 0.02
            }
        )
    
    try:
        # Preprocess
        text = ticket.text.lower().strip()
        # Vectorize
        X = vectorizer.transform([text])
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        # Get confidence
        confidence = float(max(probabilities))
        # Get all probabilities
        all_probs = {
            category: float(prob) 
            for category, prob in zip(model.classes_, probabilities)
        }
        
        return PredictionResponse(
            category=prediction,
            confidence=confidence,
            all_probabilities=all_probs
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )
