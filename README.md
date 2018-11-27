# persona.chatbot

#### almost change the code from [wildml](https://github.com/dennybritz/chatbot-retrieval/), the code implements the Dual LSTM Encoder model from [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909).

### Overview

The code here simply implements the [Personalizing Dialogue Agents: I have a dog, do you have pets too? ](https://arxiv.org/abs/1801.07243)

Use the first persona sentence as input

![image](https://github.com/fannn1217/persona.chatbot/blob/master/Images/model.jpg)

### Setup

```
Python 3
tensorflow 1.3
```

### Get the Data

from ParlAI [github](https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat)


### Training

```
python udc_train.py
```

### Evaluation

```
python udc_test.py --model_dir=...
```

#### Prediction

```
python udc_predict.py --model_dir=...
```
