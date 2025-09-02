import torch
import numpy as np
import os
from model import BiLSTMModel
from sklearn.preprocessing import MinMaxScaler

CHECKPOINT_PATH = os.path.join("checkpoints", "bilstm_model.pth")

def load_model():
    # ⚙️ مدل باید با همان تنظیمات ساخته شود
    model = BiLSTMModel(input_size=1, hidden_size=64, output_window=24, num_layers=2)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu")))
    model.eval()

    # فرض می‌کنیم فقط دما رو اسکال کردیم
    scaler = MinMaxScaler()
    return model, scaler

def predict_energy(model, scaler, input_series):
    input_series = np.array(input_series).reshape(1, -1, 1)  # [1, seq_len, 1]
    input_series_scaled = scaler.fit_transform(input_series[0])  # فرضی
    input_tensor = torch.tensor(input_series_scaled).unsqueeze(0).float()  # [1, seq_len, 1]

    with torch.no_grad():
        output = model(input_tensor)  # [1, 72]
    return output.numpy().flatten().tolist()
