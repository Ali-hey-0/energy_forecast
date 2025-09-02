import torch
import torch.nn as nn



class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_window=24):
        super(BiLSTMModel,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)

        self.fc = nn.Linear(hidden_size*2,output_window)


    def forward(self,x):
        out,_ = self.lstm(x)
        last_hidden = out[:,-1,:]
        output = self.fc(last_hidden)
        return output