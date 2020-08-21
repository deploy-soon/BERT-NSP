import os
import csv
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from transformers import *
from torch.utils.data import Dataset, DataLoader

from model import Model

years = list(map(str, range(1970, 2020))) + list(map(str, range(1, 31)))
months = ["january", "february", "march", "april", "july", "june", "august", "september", "october", "november", "december"]+years
months = months + ["year", "month", "day"]


def covert_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def get_tokenize(sentence):
        inputs = tokenizer.encode_plus(sentence, None, add_special_tokens=True, max_length=450,
                                            pad_to_max_length=True, return_token_type_ids=False)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return ids, mask
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert = nn.DataParallel(bert).cuda()
    reader = csv.DictReader(open("./evaluation.csv"))
    before_ids, before_mask, after_ids, after_mask = [], [], [], []
    for row in tqdm(reader):
        before = row["NEWS_1_BODY"]
        after = row["NEWS_2_BODY"]
        _before_ids, _before_mask = get_tokenize(before)
        _after_ids, _after_mask = get_tokenize(after)
        before_ids.append(_before_ids)
        before_mask.append(_before_mask)
        after_ids.append(_after_ids)
        after_mask.append(_after_mask)
    before_ids, before_mask = np.array(before_ids, dtype=np.int64), np.array(before_mask, dtype=np.int64)
    after_ids, after_mask = np.array(after_ids, dtype=np.int64), np.array(after_mask, dtype=np.int64)
    res_before = np.zeros((100000, 768))
    res_after = np.zeros((100000, 768))

    batch_size = 200

    with torch.no_grad():
        for i in tqdm(range(0, 100000, batch_size)):
            ids = torch.from_numpy(after_ids[i:i+batch_size]).cuda()
            mask = torch.from_numpy(after_mask[i:i+batch_size]).cuda()
            _, hidden = bert(ids, attention_mask=mask)
            res_after[i:i+batch_size] = hidden.cpu().data.numpy()

            ids = torch.from_numpy(before_ids[i:i+batch_size]).cuda()
            mask = torch.from_numpy(before_mask[i:i+batch_size]).cuda()
            _, hidden = bert(ids, attention_mask=mask)
            res_before[i:i+batch_size] = hidden.cpu().data.numpy()
    with open("before_bert_eval.npy", "wb") as fout:
        np.save(fout, res_before)
    with open("after_bert_eval.npy", "wb") as fout:
        np.save(fout, res_after)

def covert_bert_num():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def get_tokenize(sentence):
        sentence = [token for token in tokenizer.tokenize(sentence) if token in months]
        sentence = " ".join(sentence)
        inputs = tokenizer.encode_plus(sentence, None, add_special_tokens=True, max_length=20,
                                            pad_to_max_length=True, return_token_type_ids=False)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return ids, mask
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert = nn.DataParallel(bert).cuda()
    reader = csv.DictReader(open("./evaluation.csv"))
    before_ids, before_mask, after_ids, after_mask = [], [], [], []
    for row in tqdm(reader):
        before = row["NEWS_1_BODY"]
        after = row["NEWS_2_BODY"]
        _before_ids, _before_mask = get_tokenize(before)
        _after_ids, _after_mask = get_tokenize(after)
        before_ids.append(_before_ids)
        before_mask.append(_before_mask)
        after_ids.append(_after_ids)
        after_mask.append(_after_mask)
    before_ids, before_mask = np.array(before_ids, dtype=np.int64), np.array(before_mask, dtype=np.int64)
    after_ids, after_mask = np.array(after_ids, dtype=np.int64), np.array(after_mask, dtype=np.int64)
    res_before = np.zeros((100000, 768))
    res_after = np.zeros((100000, 768))

    batch_size = 200

    with torch.no_grad():
        for i in tqdm(range(0, 100000, batch_size)):
            ids = torch.from_numpy(after_ids[i:i+batch_size]).cuda()
            mask = torch.from_numpy(after_mask[i:i+batch_size]).cuda()
            _, hidden = bert(ids, attention_mask=mask)
            res_after[i:i+batch_size] = hidden.cpu().data.numpy()

            ids = torch.from_numpy(before_ids[i:i+batch_size]).cuda()
            mask = torch.from_numpy(before_mask[i:i+batch_size]).cuda()
            _, hidden = bert(ids, attention_mask=mask)
            res_before[i:i+batch_size] = hidden.cpu().data.numpy()
    with open("before_bert_years_eval.npy", "wb") as fout:
        np.save(fout, res_before)
    with open("after_bert_years_eval.npy", "wb") as fout:
        np.save(fout, res_after)


class Data(Dataset):

    def __init__(self):
        self.before_bert = np.load("before_bert_eval.npy").astype(np.float32)
        self.after_bert = np.load("after_bert_eval.npy").astype(np.float32)
        self.before_bert_year = np.load("before_bert_years_eval.npy").astype(np.float32)
        self.after_bert_year = np.load("before_bert_years_eval.npy").astype(np.float32)
        self.datalen = 100000

    def __getitem__(self, idx):
        return self.before_bert[idx], self.after_bert[idx], self.before_bert_year[idx], self.after_bert_year[idx]

    def __len__(self):
        return self.datalen


def test():
    data = Data()
    loader = DataLoader(data, batch_size=500, shuffle=False, num_workers=4)
    model = nn.DataParallel(Model())
    model.load_state_dict(torch.load("model.pt"))
    model.cuda()
    res = []

    for x1, x2, x1_t, x2_t in loader:
        x1 = x1.cuda()
        x2 = x2.cuda()
        x1_t = x1_t.cuda()
        x2_t = x2_t.cuda()
        pred = model(x1, x2, x1_t, x2_t)
        pred = list(pred.squeeze().cpu().data.numpy())
        res += pred

    fout = open("answer.csv", "w")
    writer = csv.writer(fout)
    for i, r in enumerate(res):
        writer.writerow([i+1, 1, 0] if r > 0.5 else [i+1, 0, 1])


if __name__ == "__main__":
    covert_bert()
    covert_bert_num()
    test()
