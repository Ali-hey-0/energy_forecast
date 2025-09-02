import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import BiLSTMModel
from dataset import EnergyDataset
import numpy as np 






# ⚙️ پارامترها
input_window = 168
output_window = 24
input_dim = 1
hidden_dim = 64
num_layers = 2

device = torch.device("cpu")



# 🧠 بارگذاری مدل
model = BiLSTMModel(input_dim,hidden_dim,num_layers,output_window).to(device)
model.load_state_dict(torch.load("./checkpoints/bilstm_model.pth",map_location=device))
model.eval()

# 📦 دیتاست تست
dataset = EnergyDataset("./DUQ_hourly.csv",input_window,output_window,split="test")
loader = DataLoader(dataset,batch_size=1,shuffle=False)


# 🔍 پیش‌بینی روی یک نمونه
with torch.no_grad():
    for x,y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy().flatten()
        real = y.numpy().flatten()
        
        # 📊 رسم نمودار
        plt.figure(figsize=(12,5))
        plt.plot(real,label="Real",color="blue")
        plt.plot(pred,label="Predicted",color="orange")
        plt.title("Forecasting with BiLSTM")
        plt.xlabel("Hour")
        plt.ylabel("Temperature")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        break

