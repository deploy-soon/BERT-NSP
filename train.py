import os
import csv
import torch
from transformers import *

def load():
    reader = csv.DictReader(open("./training.csv"))
    for row in reader:
        print(row)
        get_bert_emb(row["BEFORE_HEADLINE"])
        break


BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]

def get_bert_emb(sentence):
    model_class = BertModel
    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
    model = model_class.from_pretrained("bert-base-uncased")


    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
    print(input_ids)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
        print(last_hidden_states)



if __name__ == "__main__":
    load()
