# BERT-NSP

### Train Procedure
1. Tokenize each sentences and set all tokens to embedding vectors.
1. Parse only temporal words to get BERT embedding vectors.
1. Train all vectors with fully connected layers.

### How to run

```python
$ python data.py
$ python train.py
$ python inference.py
```
