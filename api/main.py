from fastapi import FastAPI
from api.predictor import load_model, predict_energy
from api.schemas import EnergyInput

app = FastAPI(title="Energy Forecast API")

# 🧠 بارگذاری مدل در استارتاپ
model, scaler = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Energy Forecast API 🚀"}

@app.post("/predict")
def predict(data: EnergyInput):
    prediction = predict_energy(model, scaler, data.values)
    return {"prediction": prediction}
