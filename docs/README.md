# BERT-NSP

## Description
BERT를 사용하기에 앞서 BERT의 결과물을 통해 다른 모델을 학습하는 방식으로 BERT 모델을 활용했습니다.본 문제는 두 문단이 주어져있을 때 두 문단의 선후관계를 파악하는 문제이다. 아래는 데이터의 예시이다.
- 이전 문단 예시
> Changes in weather affect near-term gas prices, and a mild summer this year may be followed by a warmer-than-average winter, which could pressure prices. Longer-term, the abundant supply of U.S. natural gas will require an additional demand outlet to soak up incremental output to sustain or raise prices. The likely emergence of exports of liquefied natural gas to Asia may satisfy that need. Industrial growth in the U.S. may raise natural gas consumption as well.
- 이후 문단 예시
> Domestic natural gas benchmark prices may be restrained in the near term given resilient output, in-line storage levels and expectations of less winter heating demand. Milder weather is forecast for the next eight to 14 days. While prices may show seasonal volatility, gas demand will remain restrained near-term if weather is subdued. Gas must compete for market share with coal and nuclear power. Near-term capacity constraints and weather will continue to affect winter basis differentials.

데이터는 영어로 되어 있으며 문맥 안에 시간의 정보가 있을 수도 있으며 없을 수도 있다. 두 문단을 통해 문단의 선후 관계를 예측하는 것이 본 문제의 목적이며 데이터는 총 20만 쌍이 주어져있다. 
## Model Structure
### BERT Model
문단에 시간 정보를 포함한다면 복잡한 모델 없이 문자열을 전처리하여 연월일과 같은 시간 정보를 추출하고 이를 비교하여 문단의 선후관계를 파악하면 된다. 다만 데이터에는 시간적 특성이 담겨있지 않은 문단도 다수 포함되어 있기 때문에 문맥을 파악하여 선후관계를 분석하는 작업 또한 필요하다. 문장의 문맥을 파악하기 위해서 각 단어를 임베딩 벡터로 보고 임베딩 벡터의 시간 흐름에 따라 RNN 계열의 모델을 통해 문맥을 파악할 수 있다. 최근 NLP 모델에서는 BERT를 통해 자연어를 분석하고 classfier 등 다양한 분석에 활용한다. Transformer의 encoder 부분을 활용한 BERT 모델에서 문맥 정보를 파악하기 위해 본 문제에서는 마지막 layer의 hidden state를 추출하여 분석에 활용하였다. 각 문단을 모두 BERT 모델에 넣어 활용하였고 그 결과 각 문단을 768 길이의 벡터로 변환하였다. BERT 모델은 라이브러리를 통해 가져왔으며 기본으로 pretrained 된 tokenizer와 model을 사용하였다.(라이브러리: https://github.com/huggingface/transformers)
추가로 시간적인 특성을 보다 살리기 위해 연월일과 같은 token을 미리 추출하여 그 정보만으로 문장을 재구성하고 이 문장을 다시 BERT 모델의 hidden state을 추출하였다. 이를 통해 모든 문단은 기본적인 BERT output과 시간 특성이 반영된 output을 전처리되었다.
### FC layers
Fully connected layer는 임베딩 벡터를 압축하고 각 문단의 선후관계를 파악하는 역할을 한다. 각 임베딩 벡터를 적당히 압축하고 각 쌍이 되는 임베딩 벡터는 다시 concate되어 fully connected layer를 통해 압축된다. 결과적으로 output은 1차원의 sigmoid 값이 되고 이 값이 1에 가까우면 input의 두 문장중 두번째 문장이 시간적으로 후인 것을 의미하도록 하였다.
### Train Procedure
학습 시간을 단축하기 위해 미리 BERT 모델로 모든 문단을 임베딩 벡터로 전처리하였다. 모델에는 쌍이 되는 두 문장이 같이 input으로 활용하게 되고 공평한 학습을 위해 20만개의 데이터중 10만개는 앞 문장이 시간적으로 전인 것으로 처리하여 학습하였다. Adam optimizer로 학습하였으며 loss는 binary cross entropy를 사용하였다.
## Result & Discussion
데이터중 25%를 validation set으로 하였으며 Test set은 따로 두지 않았다. 일정 epochs을 설정하고 모델의 구조를 바꾸어가며 validation loss를 비교하는 방식으로 모델을 학습하였으며 가장 좋은 모델의 validataion accuracy가 67.1%였다. 모델의 결과는 문단의 선후관계를 분석하는 것으로 최소 50%의 accuracy를 보장한다. 이에 비해 결과가 그렇게 좋지는 않았는데 그 이유로 데이터 자체가 학습하기 어려운 데이터였다고 생각한다. 직접 문단을 읽어보아도 선후관계를 파악하기 힘들었으며 학습 초기에 여러 시도를 반복하였다. 보다 학습을 개선하기 위해서 문장별 분석을 하고 문단에 attention을 주어 보다 시간적 문맥을 조건적으로 파악하는 것이 더 좋을 수 있을 것이라 생각한다.

