import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class simple_net(nn.Module):
    def __init__(self, input_size):
        super(simple_net, self).__init__()
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        ).to(torch.device("cuda"))

    def forward(self, x):
        y_ = self.fc(x)
        y = F.softmax(y_,dim=1)
        return y