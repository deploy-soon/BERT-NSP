import sys
import torch
from torch import nn
import torch.nn.init as init


class HiddenModel(nn.Module):

    def __init__(self, bert):
        super(Model, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(768, 256)
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )

    def forward(self, x1_ids, x1_mask, x2_ids, x2_mask):

        with torch.no_grad():
            _, hidden_x1 = self.bert(x1_ids, attention_mask=x1_mask)
            _, hidden_x2 = self.bert(x2_ids, attention_mask=x2_mask)
        hidden_x1 = self.fc1(hidden_x1)
        hidden_x2 = self.fc1(hidden_x2)
        hidden = torch.cat((hidden_x1, hidden_x2), dim=1)
        pred = self.fc(hidden)
        pred = torch.sigmoid(pred)
        return pred



class fc(nn.Module):

    def __init__(self, ipt, opt):
        super(fc, self).__init__()
        self.linear = nn.Linear(ipt, opt)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1_1 = fc(768, 256)
        self.fc1_2 = fc(256, 128)
        self.fc2_1 = fc(768, 256)
        self.fc2_2 = fc(256, 128)
        self.fc3 = fc(768, 128)
        self.fc4 = fc(768, 128)
        self.fc5 = fc(128*2, 128)
        self.fc6 = fc(128*2, 128)
        self.fc = nn.Sequential(
            fc(128 * 2, 128),
            fc(128, 32),
            fc(32, 8),
            nn.Linear(8, 1)
        )

    def forward(self, x1, x2, x1_t, x2_t):
        x1 = self.fc1_1(x1)
        x1 = self.fc1_2(x1)
        x2 = self.fc2_1(x2)
        x2 = self.fc2_2(x2)
        x1_t = self.fc3(x1_t)
        x2_t = self.fc4(x2_t)

        sentence = torch.cat((x1, x2), dim=1)
        sentence = self.fc5(sentence)
        years = torch.cat((x1_t, x2_t), dim=1)
        years = self.fc6(years)

        hidden = torch.cat((sentence, years), dim=1)
        pred = self.fc(hidden)
        pred = torch.sigmoid(pred)
        return pred
