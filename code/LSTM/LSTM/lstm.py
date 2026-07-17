import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Phase_classifier(nn.Module):
    def __init__(self,input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.rnn=nn.LSTM(input_size, hidden_size=128, batch_first=True)
        self.fc1=nn.Linear(128, 64)
        self.fc2=nn.Linear(64, num_classes)
                                                           
    def forward(self, x):
    #    x_trans=x.view(x.size(0), x.size(1), self.input_size)
       lstm_out, temp = self.rnn(x)
       lstm_out=lstm_out[:,-1,:]
       lstm_out = self.fc1(lstm_out)
       lstm_out = F.relu(lstm_out)
       lstm_out = self.fc2(lstm_out)

    #    pred = F.log_softmax(lstm_out, dim=1)
    #    pred=torch.argmax(pred,dim=1)
       return lstm_out
     