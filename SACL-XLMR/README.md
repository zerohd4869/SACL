# SACL-XLMR

This repository contains the official code for the paper [UCAS-IIE-NLP at SemEval-2023 Task 12: Enhancing Generalization of Multilingual BERT for Low-resource Sentiment Analysis](https://aclanthology.org/2023.semeval-1.255.pdf), which has been accepted by [SemEval@ACL 2023].

## Introduction
We propose SACL-XLMR, a multilingual system for sentiment analysis on low-resource African languages. Our system outperformed the comparison baselines on both multilingual and zero-shot sentiment classification subtasks, and obtained the 1st rank on the zero-shot classification subtask in the official ranking. 

SACL-XLMR employs a lexicon-based multilingual BERT to facilitate language adaptation and sentiment-aware representation learning. It also uses a supervised adversarial contrastive learning technique to learn sentiment-spread structured representations and enhance model generalization.

## Result Reproduction for AfriSenti-SemEval Task C 

1. Clone the repository and download related data and models
```
git clone https://github.com/zerohd4869/SACL.git
cd /SACL/SACL-XLMR
```
Download the `afro-xlmr-large` model parameters from [here](https://huggingface.co/Davlan/afro-xlmr-large) and place them in the `/SACL/ptms/afro-xlmr-large/` directory. 
Then, download the related data from [here](https://drive.google.com/file/d/1c6TwpVDoH1Uj-0E7L0BWNxaKW2WZlwfL/view?usp=share_link), and place the unzip files in the `/SACL/SACL-XLMR/afrisent-semeval-2023/` directory.
And download the best SACL-XLMR model parameters from [here](https://drive.google.com/file/d/17bBEuUfKiOgaIwafQz5eyBOzzLPRIjow/view?usp=sharing), and place the unzip files in the `/SACL/SACL-XLMR/sacl_xlmr_best_models/` directory.
Please note that we only provide the best 1-fold model parameters for SACL-XLMR due to cloud storage capacity limitation.

2. Install dependencies
``` 
# env: Python 3.7.16, Tesla V100 32GB
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

If you are interested in this work and want to use the code in this repo, please **star** this repo and **cite** by:

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