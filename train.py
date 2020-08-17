import os
import csv
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import *

from model import Model
from data import Data


class Train:

    def __init__(self, batch_size=32, learning_rate=0.0005, epochs=10):

        self.epochs = epochs
        self.learing_rate = learning_rate
        self.batch_size = batch_size

        self.data = Data()
        self.train_num = int(len(self.data) * 0.75)
        self.vali_num = len(self.data) - self.train_num
        train_set, vali_set = random_split(self.data, [self.train_num , self.vali_num])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        self.vali_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)

        self.model = nn.DataParallel(Model())
        self.model = self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss(reduction="sum")

    def train(self):
        self.model.train()
        train_loss = 0
        for x1, x2, y in self.train_loader:
            pred = self.model(x1, x2)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / self.train_num

    def vali(self):
        self.model.eval()
        vali_loss = 0
        for x1, x2, y in self.train_loader:
            pred = self.model(x1, x2)
            loss = self.criterion(pred, y)
            vali_loss += loss.item()
        return vali_loss / self.vali_num

    def run(self):
        for epoch in range(self.epochs):
            train_loss = self.train()
            vali_loss = self.vali()
            print(epoch, train_loss, vali_loss)


if __name__ == "__main__":
    train = Train()
    train.run()
