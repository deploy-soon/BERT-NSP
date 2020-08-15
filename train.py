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

    inputs = tokenizer.encode_plus(
        sentence,
        None,
        add_special_tokens=True,
        max_length=200,
        pad_to_max_length=True,
        return_token_type_ids=True
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]

    input_ids = torch.tensor([ids], dtype=torch.long)
    input_mask = torch.tensor([mask], dtype=torch.long)
    input_token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
    print(input_ids.size())
    print(input_ids)
    print(input_mask.size())
    print(input_mask)
    print(input_token_type_ids.size())
    print(input_token_type_ids)
    with torch.no_grad():
        _, last_hidden_states = model(input_ids, attention_mask=input_mask, token_type_ids=input_token_type_ids)
        print(last_hidden_states)
        print(last_hidden_states.size())



if __name__ == "__main__":
    load()
