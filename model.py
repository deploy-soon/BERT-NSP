import sys
import torch
from torch import nn
import torch.nn.init as init


class Model(nn.Module):

    def __init__(self, bert):
        super(Model, self).__init__()
        self.bert = bert
        self.fc = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )

    def forward(self, x1_ids, x1_mask, x1_type, x2_ids, x2_mask, x2_type):

        with torch.no_grad():
            _, hidden_x1 = self.bert(x1_ids, attention_mask=x1_mask, token_type_ids=x1_type)
            _, hidden_x2 = self.bert(x2_ids, attention_mask=x2_mask, token_type_ids=x2_type)
        hidden = torch.cat((hidden_x1, hidden_x2), dim=1)
        pred = self.fc(hidden)
        pred = torch.sigmoid(pred)
        return pred
