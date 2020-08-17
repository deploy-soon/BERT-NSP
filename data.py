import os
import csv
from tqdm import tqdm
import torch
from transformers import *
from torch.utils.data import Dataset, DataLoader


BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForTokenClassification,
                      BertForQuestionAnswering]


class Data(Dataset):

    def __init__(self, model_cls=BertModel, size=200):

        self.model = model_cls.from_pretrained("bert-base-uncased")
        self.model = nn.DataParallel(self.model).to(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        reader = csv.DictReader(open("./training.csv"))
        xs, ys = dict(), dict()
        datalen = 0
        for row in tqdm(reader):
            x = self.get_bert_emb(row["BEFORE_HEADLINE"])
            y = self.get_bert_emb(row["AFTER_HEADLINE"])
            idx = int(row["INDEX"])
            xs[idx], ys[idx] = x, y
            datalen = datalen + 1
            if size <= datalen:
                break
        self.datalen = datalen
        self.xs = xs
        self.ys = ys

    def get_bert_emb(self, sentence):

        inputs = self.tokenizer.encode_plus(sentence, None, add_special_tokens=True, max_length=200,
                                            pad_to_max_length=True, return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        input_ids = torch.tensor([ids], dtype=torch.long)
        input_mask = torch.tensor([mask], dtype=torch.long)
        input_token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
        with torch.no_grad():
            _, hidden_states = self.model(input_ids, attention_mask=input_mask, token_type_ids=input_token_type_ids)
            hidden_states = hidden_states.cpu().data.numpy()
            return hidden_states

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

if __name__ == "__main__":
    data = Data()
    loader = DataLoader(data, batch_size=2)
    for xs, ys, in loader:
        print(xs.size())
        print(ys.size())
        break
