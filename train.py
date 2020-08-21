import os
import csv
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import *

from model import Model
from data import Data


class Train:

    def __init__(self, batch_size=256* 3, learning_rate=0.0004, epochs=800):

        self.epochs = epochs
        self.learing_rate = learning_rate
        self.batch_size = batch_size

        self.data = Data()
        self.train_num = int(len(self.data) * 0.75)
        self.vali_num = len(self.data) - self.train_num
        train_set, vali_set = random_split(self.data, [self.train_num , self.vali_num])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        self.vali_loader = DataLoader(vali_set, batch_size=batch_size, num_workers=4)

        self.model = nn.DataParallel(Model())
        self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss(reduction="sum")

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        for x1, x2, x1_t, x2_t, y in tqdm(self.train_loader):
            x1 = x1.cuda()
            x2 = x2.cuda()
            x1_t = x1_t.cuda()
            x2_t = x2_t.cuda()
            y = y.to("cuda", dtype=torch.float32)
            y = y.view(-1, 1)
            pred = self.model(x1, x2, x1_t, x2_t)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            correct += torch.sum((pred > 0.5) == y).item()
        return train_loss / self.train_num, correct / self.train_num

    def vali(self):
        self.model.eval()
        vali_loss = 0
        correct = 0
        for x1, x2, x1_t, x2_t, y in self.vali_loader:
            x1 = x1.cuda()
            x2 = x2.cuda()
            x1_t = x1_t.cuda()
            x2_t = x2_t.cuda()
            y = y.to("cuda", dtype=torch.float32)
            y = y.view(-1, 1)
            pred = self.model(x1, x2, x1_t, x2_t)
            loss = self.criterion(pred, y)
            vali_loss += loss.item()
            correct += torch.sum((pred > 0.5) == y).item()
        return vali_loss / self.vali_num, correct / self.vali_num

    def run(self):
        max_loss = 999
        for epoch in range(self.epochs):
            train_loss = self.train()
            vali_loss = self.vali()
            print(epoch, train_loss, vali_loss)
            if vali_loss[0] < max_loss:
                max_loss = vali_loss[0]
                torch.save(self.model.state_dict(), "model.pt")



if __name__ == "__main__":
    train = Train()
    train.run()
