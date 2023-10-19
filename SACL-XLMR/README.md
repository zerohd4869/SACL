# SACL-XLMR

This repository provides the official repository of the paper [UCAS-IIE-NLP at SemEval-2023 Task 12: Enhancing Generalization of
Multilingual BERT for Low-resource Sentiment Analysis](https://aclanthology.org/2023.semeval-1.255.pdf) (**Accepted by [SemEval@ACL 2023].


## Introduction
In this work, we propose a multilingual system named SACL-XLMR for sentiment analysis on low-resource African languages. The system achieved competitive results, largely outperforming the comparison baselines on both multilingual and zero-shot sentiment classification subtasks, and obtained
the 1st rank on zero-shot classification subtask in the official ranking.

Specifically, the system employs a lexicon-based multilingual BERT to facilitate language adaptation and sentiment-aware representation learning. It also uses a supervised adversarial contrastive learning technique to learn sentiment-spread structured representations and enhance model generalization. 

## Result Reproduction for AfriSenti-SemEval Task C 

1. Clone the repository and download related models
```
git clone https://github.com/zerohd4869/SACL.git
cd SACL/SACL-XLMR
```
Download the `afro-xlmr-large` model parameters from [here](https://huggingface.co/Davlan/afro-xlmr-large) and place them in the `/SACL/ptms/afro-xlmr-large/` directory. Download the best SACL-XLMR model parameters from [here](https://drive.google.com/file/d/17bBEuUfKiOgaIwafQz5eyBOzzLPRIjow/view?usp=sharing), extract them, and place them in the `/SACL/SACL-XLMR/sacl_xlmr_best_models/` directory.
Notably, due to cloud storage capacity limitation, we only provide the best 1-fold model parameters for SACL-XLMR.

2. Install related dependencies
```
# Python 3.7.16, Tesla V100 32GB
pip install -r sacl_xlmr_requirements.txt
```

3. Run examples
```
# Tigrinya track
nohup bash run_main_t12_tg.sh >  run_main_t12_tg.out &

# Oromo track
nohup bash run_main_t12_or.sh >  run_main_t12_or.out &
```


## Citation

If you are interested in this work, and want to use the codes in this repo, please **star** this repo and **cite** by:


```
@inproceedings{DBLP:conf/semeval/0001WLZH23,
  author       = {Dou Hu and
                  Lingwei Wei and
                  Yaxin Liu and
                  Wei Zhou and
                  Songlin Hu},
  title        = {{UCAS-IIE-NLP} at SemEval-2023 Task 12: Enhancing Generalization of
                  Multilingual {BERT} for Low-resource Sentiment Analysis},
  booktitle    = {SemEval@ACL},
  pages        = {1849--1857},
  publisher    = {Association for Computational Linguistics},
  year         = {2023}
}
```