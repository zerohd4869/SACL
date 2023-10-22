# SACL-LSTM

This repository provides the official repository of the paper [Supervised Adversarial Contrastive Learning for Emotion Recognition in Conversations](https://aclanthology.org/2023.acl-long.606.pdf) (**Accepted by [ACL 2023]**).

## Introduction
In this work, we present a supervised adversarial contrastive learning (SACL) framework for learning class-spread structured representations in a supervised manner. 
Under the framework with contextual adversarial training, we develop a sequence-based SACL-LSTM to learn label-consistent and context-robust features for ERC task.

## Quick Start

1. Clone the repository and download related data features
```
git clone https://github.com/zerohd4869/SACL.git
cd /SACL/SACL-LSTM
```

2. Install dependencies
``` 
# env: Python 3.7.16, Tesla V100 32GB
pip install -r sacl_lstm_requirements.txt
```

3. Run examples
```
# IEMOCAP dataset
nohup bash script/run_train_bert_ie.sh >  sacl_lstm_bert_ie.out &

# MELD dataset
nohup bash script/run_train_bert_me.sh >  sacl_lstm_bert_me.out &

# EmoryNLP dataset
nohup bash script/run_train_bert_emo.sh >  sacl_lstm_bert_emo.out &

```


## Result Reproduction

1. Clone the repository and download related features and models
```
git clone https://github.com/zerohd4869/SACL.git
cd /SACL/SACL-LSTM
```

2. Install dependencies
``` 
# env: Python 3.7.16, Tesla V100 32GB
pip install -r sacl_lstm_requirements.txt
```

3. Run examples
```
# Three dataset: IEMOCAP, MELD and EmoryNLP
nohup bash script/run_train_bert_inference.sh >  sacl_lstm_bert_inference.out &

```


## Citation

If you are interested in this work, and want to use the codes in this repo, please **star** this repo and **cite** by:


```
@inproceedings{DBLP:conf/acl/0001BWZH23,
  author       = {Dou Hu and
                  Yinan Bao and
                  Lingwei Wei and
                  Wei Zhou and
                  Songlin Hu},
  title        = {Supervised Adversarial Contrastive Learning for Emotion Recognition
                  in Conversations},
  booktitle    = {{ACL} {(1)}},
  pages        = {10835--10852},
  publisher    = {Association for Computational Linguistics},
  year         = {2023}
}
```