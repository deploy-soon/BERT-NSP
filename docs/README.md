# BERT-NSP

## Description
BERT를 사용하기에 앞서 BERT의 결과물을 통해 다른 모델을 학습하는 방식으로 BERT 모델을 활용했습니다.본 문제는 두 문단이 주어져있을 때 두 문단의 선후관계를 파악하는 문제이다. 아래는 데이터의 예시이다.
- 이전 문단 예시
> Changes in weather affect near-term gas prices, and a mild summer this year may be followed by a warmer-than-average winter, which could pressure prices. Longer-term, the abundant supply of U.S. natural gas will require an additional demand outlet to soak up incremental output to sustain or raise prices. The likely emergence of exports of liquefied natural gas to Asia may satisfy that need. Industrial growth in the U.S. may raise natural gas consumption as well.
- 이후 문단 예시
> Domestic natural gas benchmark prices may be restrained in the near term given resilient output, in-line storage levels and expectations of less winter heating demand. Milder weather is forecast for the next eight to 14 days. While prices may show seasonal volatility, gas demand will remain restrained near-term if weather is subdued. Gas must compete for market share with coal and nuclear power. Near-term capacity constraints and weather will continue to affect winter basis differentials.

데이터는 영어로 되어 있으며 문맥 안에 시간의 정보가 있을 수도 있으며 없을 수도 있다. 두 문단을 통해 문단의 선후 관계를 예측하는 것이 본 문제의 목적이며 데이터는 총 20만 쌍이 주어져있따.
## Model Structure
## Result & Discussion

