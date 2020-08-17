import sys
import torch
from torch import nn
import torch.nn.init as init


class Model(nn.Module):

    def __init__(self, classes):
        super(Model, self).__init__()
        self.base = nn.Sequential()

    def forward(self, x, y):
        pass
