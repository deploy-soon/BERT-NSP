import os
import csv
import numpy as np
from tqdm import tqdm
import torch
from transformers import *
from torch.utils.data import Dataset, DataLoader



BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForTokenClassification,
                      BertForQuestionAnswering]


class Data(Dataset):

    def __init__(self, model_cls=BertModel):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        reader = csv.DictReader(open("./training.csv"))
        x1_ids, x1_mask, x1_type = dict(), dict(), dict()
        x2_ids, x2_mask, x2_type = dict(), dict(), dict()
        ys = []
        datalen = 0
        ids_list, mask_list, type_list = [], [], []
        for row in tqdm(reader):
            idx = int(row["INDEX"])
            if idx % 2 == 0:
                x1_ids[idx], x1_mask[idx], x1_type[idx] = self.get_tokenize(row["BEFORE_HEADLINE"])
                x2_ids[idx], x2_mask[idx], x2_type[idx] = self.get_tokenize(row["AFTER_HEADLINE"])
                ys.append([1])
            else:
                x1_ids[idx], x1_mask[idx], x1_type[idx] = self.get_tokenize(row["AFTER_HEADLINE"])
                x2_ids[idx], x2_mask[idx], x2_type[idx] = self.get_tokenize(row["BEFORE_HEADLINE"])
                ys.append([0])

            if idx > 20000:
                break
        x1_ids = {k: np.array(v, dtype=np.int64) for k, v in x1_ids.items()}
        x1_mask = {k: np.array(v, dtype=np.int64) for k, v in x1_mask.items()}
        x1_type = {k: np.array(v, dtype=np.int64) for k, v in x1_type.items()}
        x2_ids = {k: np.array(v, dtype=np.int64) for k, v in x2_ids.items()}
        x2_mask = {k: np.array(v, dtype=np.int64) for k, v in x2_mask.items()}
        x2_type = {k: np.array(v, dtype=np.int64) for k, v in x2_type.items()}
        self.x1_ids, self.x1_mask, self.x1_type = x1_ids, x1_mask, x1_type
        self.x2_ids, self.x2_mask, self.x2_type = x2_ids, x2_mask, x2_type
        self.ys = np.array(ys, dtype=np.float32)
        self.datalen = len(list(x1_ids.keys()))

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        return self.x1_ids[idx], self.x1_mask[idx], self.x1_type[idx],\
            self.x2_ids[idx], self.x2_mask[idx], self.x2_type[idx], self.ys[idx]

    def get_tokenize(self, sentence):

        inputs = self.tokenizer.encode_plus(sentence, None, add_special_tokens=True, max_length=200,
                                            pad_to_max_length=True, return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids, mask, token_type_ids


if __name__ == "__main__":
    data = Data()
    loader = DataLoader(data, batch_size=2, num_workers=4)
    for xs, _, _, ys, _, _, _ in loader:
        print(xs.size())
        print(ys.size())
        break
