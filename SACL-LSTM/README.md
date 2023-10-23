# SACL-LSTM

This repository provides the official code for the paper [Supervised Adversarial Contrastive Learning for Emotion Recognition in Conversations](https://arxiv.org/pdf/2306.01505.pdf) (**Accepted by [ACL 2023]**).

## Introduction
The work presents a supervised adversarial contrastive learning (SACL) framework for learning class-spread structured representations in a supervised manner. 
Under the framework with contextual adversarial training, it also introduces a sequence-based SACL-LSTM model that learns label-consistent and context-robust features for the ERC task.

## Quick Start

1. Clone the repository and extract data features
```
git clone https://github.com/zerohd4869/SACL.git
cd /SACL/SACL-LSTM
```

To extract the roberta features for the target dataset, follow the scripts in the [COSMIC](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC/feature-extraction) repo and place them in the `/SACL/SACL-LSTM/data/` directory.


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

The original datasets can be found at [IEMOCAP](https://sail.usc.edu/iemocap/), [SEMAINE](https://semaine-db.eu), and [MELD](https://github.com/SenticNet/MELD). 
Download the processed data features with `roberta-large` embedding from [here](https://drive.google.com/file/d/1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr/view) and place them in the `/SACL/SACL-LSTM/data/` directory. 
Then, download the best SACL-LSTM model parameters from [here](https://drive.google.com/file/d/1TRDeo6speGlmQ5tmV7Jv6NSwg-Pyw4Iv/view?usp=sharing), unzip the file, and place them in the `/SACL/SACL-LSTM/sacl_lstm_best_models/` directory.


2. Install dependencies
``` 
# env: Python 3.7.16, Tesla V100 32GB
pip install -r sacl_lstm_requirements.txt
```

3. Run examples
```
# Three datasets: IEMOCAP, MELD, and EmoryNLP
nohup bash script/run_train_bert_inference.sh >  sacl_lstm_bert_inference.out &

```


## Citation

If you are interested in this work and want to use the code in this repository, please **star** this repo and **cite** it as follows:


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
