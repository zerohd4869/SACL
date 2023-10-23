# SACL-LSTM

This repository provides the official code for the paper [Supervised Adversarial Contrastive Learning for Emotion Recognition in Conversations](https://arxiv.org/pdf/2306.01505.pdf) (**Accepted by [ACL 2023]**).

## Introduction
The work presents a supervised adversarial contrastive learning (SACL) framework for learning class-spread structured representations in a supervised manner. 
Under the framework with contextual adversarial training, a sequence-based SACL-LSTM model is introduced to learn label-consistent and context-robust features for the ERC task.

## Quick Start

1. Clone the repository and extract data features
```
git clone https://github.com/zerohd4869/SACL.git
cd /SACL/SACL-LSTM
```

Extract the roberta features for the target dataset following the scripts in the [COSMIC](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC/feature-extraction) repo, and place them in the `/SACL/SACL-LSTM/data/` directory.


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

The original datasets can be found at [IEMOCAP](https://sail.usc.edu/iemocap/), [MELD](https://github.com/SenticNet/MELD), and [EmoryNLP](https://github.com/emorynlp/character-mining). 
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
bash script/run_train_bert_inference.sh &

```

4. Results of SACL-LSTM

IEMOCAP dataset:

|Model |Happy|Sad|Neutral|Angry|Excited|Frustrated|*Acc*|*Macro-F1*|*Weighted-F1*|
|:----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**SACL-LSTM (paper, the average result with five random seeds)** |56.91|84.78|70.00|64.09|69.70|65.02|69.08|68.42|69.22|
|SACL-LSTM (the result with a best seed) |58.72|84.85|70.96|64.67|71.27|63.87|69.62|69.06|69.70|


MELD dataset:

|Model |Neutral|Surprise|Fear|Sad|Happy|Disgust|Anger|*Acc*|*Macro-F1*|*Weighted-F1*|
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**SACL-LSTM (paper, the average result with five random seeds)** |80.17|58.77|26.23|41.34|64.98|31.47|52.35|67.51|50.76|66.45|
|SACL-LSTM (the result with a best seed) |80.30|59.66|28.57|41.46|65.25|31.19|53.55|67.89|51.43|66.86|

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
