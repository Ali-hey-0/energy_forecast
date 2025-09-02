import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EnergyDataset
from model import BiLSTMModel
import os

# ⚙️ هایپرپارامترها
input_window = 168
output_window = 24
batch_size = 32
num_epochs = 7
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 📦 دیتاست و DataLoader
dataset = EnergyDataset("./DUQ_hourly.csv",input_window,output_window,split="train")
train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)


# 🧠 مدل
model =  BiLSTMModel(input_size=1,hidden_size=64,num_layers=2,output_window=output_window).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




# 📁 ذخیره‌سازی
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir,exist_ok=True)
model_path = os.path.join(checkpoint_dir,"bilstm_model.pth")



# 🔁 حلقه آموزش

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x,y in train_loader:
         x = x.to(device)  # [B, T] → [B, T, 1]
         y = y.to(device)
         optimizer.zero_grad()
         output = model(x)
         loss = criterion(output,y)
         loss.backward()
         optimizer.step()
         total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")



# 💾 ذخیره مدل

torch.save(model.state_dict(),model_path)
print(f"✅ Model saved to: {model_path}")