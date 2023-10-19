#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import csv

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

import gc
import re
import os
import math
import argparse
import numpy as np
import logging.handlers
from utils import function_utils
from importlib import import_module
from transformers import AutoTokenizer
from utils.function_utils import init_logger

import torch
import torch.nn.functional as F
from torch.utils.data import (Dataset, DataLoader, SequentialSampler, RandomSampler)
from torch.utils.data import WeightedRandomSampler

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

gc.enable()
import warnings

warnings.filterwarnings('ignore')
try:
    from apex import amp

    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False
print(f"Apex AMP Installed : {APEX_INSTALLED}")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def clean_text(text, clean_flag=False):
    # text = text.lower()
    text = text.replace('""""""""""""""""', '""')
    k = 4
    while k > 1:
        while "@user " * k in text:
            text = text.replace("@user " * k, "@user ")
        k = int(k // 2)

    # TODO 4-1 no clean
    if clean_flag:
        # Twitter handle
        text = re.sub('@[\w]*', '[user]', str(text))
        text = re.sub(r'(https?|ftp|file)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[url]', text, flags=re.MULTILINE)
        text = re.sub(r'(https?|ftp|file)?:\\/\\/(\w|\.|\\/|\?|\=|\&|\%)*\b', '[url]', text, flags=re.MULTILINE)
        text = re.sub(r'(https?|ftp|file)?\\/\\/(\w|\.|\\/|\?|\=|\&|\%)*\b', '[url]', text, flags=re.MULTILINE)
        text = re.sub(r'(https?|ftp|file)?:\\/\\/', '[url]', text, flags=re.MULTILINE)
        text = re.sub(r'(https?|ftp|file)?:', '[url]', text, flags=re.MULTILINE)

        text = re.sub('\n', '', text)

        # Add some acronym substitutions
        text = re.sub(' u ', ' you ', text)
        text = re.sub(' ur', ' your', text)
        text = re.sub('btw', 'by the way', text)
        text = re.sub('gosh', 'god', text)
        text = re.sub('omg', 'oh my god', text)
        text = re.sub(' 4 ', ' for ', text)
        text = re.sub('sry', 'sorry', text)
        text = re.sub('idk', 'i do not know', text)

        text = re.sub(r"’s", " is", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"cannot", " can not", text)
        text = re.sub(r"ca 't", " can not", text)
        text = re.sub(r"wo n\'t", " can not", text)
        text = re.sub(r"wo n't", " can not", text)
        text = re.sub(r"what\'s", " what is", text)
        text = re.sub(r"What\'s", " What is", text)
        text = re.sub(r"what 's", " what is", text)
        text = re.sub(r"What 's", " what is", text)
        text = re.sub(r"how 's", " how is", text)
        text = re.sub(r"How 's", " How is", text)
        text = re.sub(r"how \'s", " how is", text)
        text = re.sub(r"How \'s", " How is", text)
        text = re.sub(r"it \'s", " it is", text)
        text = re.sub(r"it \'s", " it is", text)
        text = re.sub(r"i\'m", "i am", text)
        text = re.sub(r"I\'m", "i am", text)
        text = re.sub(r"i \'m", "i am", text)
        text = re.sub(r"i’m", "i am", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"\'d", "would", text)
        text = re.sub(r"'d", "would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"'ve ", " have ", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n’t", " not", text)
        text = re.sub(r"’ll", " will", text)
        text = re.sub(r"3m , h&amp:m and c&amp", " ", text)
        text = re.sub(r"&amp: #x27 : s", " ", text)
        text = re.sub(r"at&amp:", " ", text)
        text = re.sub(r"q&amp", " ", text)
        text = re.sub(r"&amp", " ", text)
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" fb ", "facebook", text)
        text = re.sub(r"facebooks", "facebook", text)
        text = re.sub(r"facebooking", "facebook", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r"dollars", " dollar ", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r" u\.n\.", "un", text)
        text = re.sub(r" c\.i\.a\.", "cia", text)
        text = re.sub(r" d\.c\.", "dc", text)
        text = re.sub(r" n\.j\.", "nj", text)
        text = re.sub(r" f\.c\.", "fc", text)
        text = re.sub(r" h\.r\.", "hr", text)
        text = re.sub(r" l\.a\.", "la", text)
        text = re.sub(r" u\.k\.", "uk", text)
        text = re.sub(r" p\.f\.", "pf", text)
        text = re.sub(r" h\.w\.", "hw", text)
        text = re.sub(r" n\.f\.l\.", "nfl", text)
        text = re.sub(r"'", "", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", "", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r" : ", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)
        text = re.sub(r"&", "and", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"\$", " dollar ", text)
        text = re.sub(r" [b-hB-H] ", " ", text)
        text = re.sub(r" [J-ZJ-Z] ", " ", text)
        text = re.sub(r"oooo", " ", text)
        text = re.sub(r"#", " #", text)

    from collections import defaultdict
    input_dict = defaultdict(list)
    text_l = text.lower()
    for k, v in emo_dict.items():
        if k in text_l:
            input_dict[v.lower()].append(k)
    if len(input_dict) > 0:
        input_pre = ""
        for k, v in input_dict.items():
            input_pre += k + " words: " + ",".join(v) + ". "
        text = input_pre + " </s> </s>  " + text
    return text


def proprecess_data(data, mode):
    # id, text, keyword, country_code, labels
    rows = []
    for i in range(data.shape[0]):
        temp_text = data["text"][i]
        temp_keyword = data["keyword"][i]
        start_p = temp_text.find(temp_keyword)
        if start_p == -1:
            temp_keyword = " ".join(temp_keyword.split("-"))
            start_p = temp_text.find(temp_keyword)
            if start_p < 0:
                break
        end_p = start_p + len(temp_keyword)
        if start_p > 0:
            while temp_text[start_p] != " ":
                start_p -= 1
        else:
            start_p = 0
        if end_p < len(temp_text) - 1:
            while temp_text[end_p] != " " and end_p < len(temp_text) - 1:
                end_p += 1
        else:
            end_p = len(temp_text) - 1

        text = temp_text[:start_p] + " <e> " + temp_text[start_p + 1:end_p] + " <e/>" + temp_text[end_p:]
        if mode == "train":
            rows.append(
                {'text': text,
                 'keyword': data["keyword"][i],
                 'country_code': data["country_code"][i],
                 'labels': data["labels"][i],
                 'par_id': data["par_id"][i]}
            )

        if mode == "test":
            rows.append(
                {'text': text,
                 'keyword': data["keyword"][i],
                 'country_code': data["country_code"][i],
                 'par_id': data["par_id"][i]}
            )
    train_df = pd.DataFrame(rows)
    return train_df


def read_dataset(la='ha'):
    # emo_dict
    global emo_dict
    if la == 'ha':
        # words,sentiment
        neg_dict_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/{la}/hausa_negative.csv", delimiter=",", encoding='utf-8')
        pos_dict_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/{la}/hausa_positive.csv", delimiter=",", encoding='utf-8')
        emo_dict_df = pd.concat([neg_dict_df, pos_dict_df])
        emo_dict_df.set_index('words', inplace=True)
        emo_dict = emo_dict_df['sentiment'].to_dict()
    elif la == 'yo':
        # Words,Sentiment
        neg_dict_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/{la}/yoruba_negative.csv", delimiter=",", encoding='utf-8')
        pos_dict_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/{la}/yoruba_positive.csv", delimiter=",", encoding='utf-8')
        emo_dict_df = pd.concat([neg_dict_df, pos_dict_df])
        emo_dict_df.set_index('Words', inplace=True)
        emo_dict = emo_dict_df['Sentiment'].to_dict()
    elif la == 'ig':
        # Word,Sentiment
        neg_dict_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/ig/igbo_negative.csv", delimiter=",", encoding='utf-8')
        pos_dict_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/ig/igbo_positive.csv", delimiter=",", encoding='utf-8')
        emo_dict_df = pd.concat([neg_dict_df, pos_dict_df])
        emo_dict_df.set_index('Word', inplace=True)
        emo_dict = emo_dict_df['Sentiment'].to_dict()
    elif la == 'kr':
        # Track 9: kr, afrisent-semeval-2023/sentiment_lexicon/Kinyarwanda/Kinyarwanda-NRC-EmoLex.txt
        # ['English Word', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'Kinyarwanda Word']
        dict_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/Kinyarwanda/Kinyarwanda-NRC-EmoLex.txt", delimiter="\t", encoding='utf-8')
        neg_list = list(dict_df[dict_df.negative == 1]['Kinyarwanda Word'])
        pos_list = list(dict_df[dict_df.positive == 1]['Kinyarwanda Word'])
        all_key_list = neg_list + pos_list
        all_value_list = ['negative' for _ in range(len(neg_list))] + ['positive' for _ in range(len(pos_list))]
        emo_dict = dict(zip(all_key_list, all_value_list))
    elif la == 'twi':
        # Track 10: twi, afrisent-semeval-2023/sentiment_lexicon/Twi
        # ['word', 'Unnamed: 1']
        # excel tools: et-xmlfile-1.1.0 openpyxl-3.0.10
        neg_dict_df = pd.read_excel(f"afrisent-semeval-2023/sentiment_lexicon/Twi/twi_affin_translated_neg.xlsx")
        neg_dict_df['sentiment'] = 'negative'
        neg_dict_df = neg_dict_df.drop('word', axis=1)
        neg_dict_df.set_index('Unnamed: 1', inplace=True)
        pos_dict_df = pd.read_excel(f"afrisent-semeval-2023/sentiment_lexicon/Twi/twi_affin_translated_pos.xlsx", header=None)
        pos_dict_df['sentiment'] = 'positive'
        pos_dict_df.set_index(0, inplace=True)

        dict_df = pd.concat([neg_dict_df, pos_dict_df])
        emo_dict = dict_df['sentiment'].to_dict()
    elif la == 'ma':
        # Track 7: ma, afrisent-semeval-2023/sentiment_lexicon/Darija
        # Index(['n1', 'n2', 'n3', 'n4', 'eng'], dtype='object')
        adj_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/Darija/adjectives.txt", delimiter=",", encoding='utf-8')
        emo_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/Darija/emotions.csv", delimiter=",", encoding='utf-8')
        all_df = pd.concat([adj_df, emo_df])
        all_df = all_df.melt(id_vars=['eng'], value_name='words').dropna(axis=0)

        eng2sent_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/Kinyarwanda/Kinyarwanda-NRC-EmoLex.txt", delimiter="\t", encoding='utf-8')
        # ['English Word', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'Kinyarwanda Word']
        eng2sent_df = eng2sent_df[['English Word', 'negative', 'positive']]
        eng2sent_df.rename(columns={'English Word': 'eng'}, inplace=True)

        merge_df = pd.merge(all_df, eng2sent_df, how='left', on='eng')
        neg_list = list(merge_df[merge_df.negative == 1]['words'])
        pos_list = list(merge_df[merge_df.positive == 1]['words'])
        all_key_list = neg_list + pos_list
        all_value_list = ['negative' for _ in range(len(neg_list))] + ['positive' for _ in range(len(pos_list))]
        emo_dict = dict(zip(all_key_list, all_value_list))

    elif la == 'or':
        # taskc afrisent-semeval-2023/sentiment_lexicon/om
        neg_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/om/oro_lexicon_neg.txt", delimiter=",", encoding='utf-8', header=None)
        neg_df['sentiment'] = 'negative'
        neg_df.set_index(0, inplace=True)

        pos_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/om/oro_lexicon_pos.txt", delimiter=",", encoding='utf-8', header=None)
        pos_df['sentiment'] = 'positive'
        pos_df.set_index(0, inplace=True)

        dict_df = pd.concat([pos_df, pos_df])
        emo_dict = dict_df['sentiment'].to_dict()

    elif la == 'tg':
        # taskc afrisent-semeval-2023/sentiment_lexicon/ti

        neg_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/ti/tig_lexicon_neg.txt", delimiter=",", encoding='utf-8', header=None)
        neg_df['sentiment'] = 'negative'
        neg_df.set_index(0, inplace=True)

        pos_df = pd.read_csv(f"afrisent-semeval-2023/sentiment_lexicon/ti/tig_lexicon_pos.txt", delimiter=",", encoding='utf-8', header=None)
        pos_df['sentiment'] = 'positive'
        pos_df.set_index(0, inplace=True)

        dict_df = pd.concat([pos_df, pos_df])
        emo_dict = dict_df['sentiment'].to_dict()
    else:
        emo_dict = {}

    print("emo_dict len: ", len(emo_dict))


    task_dev_df = pd.read_csv(f"afrisent-semeval-2023/SubtaskC/dev_gold/{la}_dev_gold_label.tsv", delimiter="\t", encoding='utf-8', quoting=csv.QUOTE_NONE)
    task_dev_df = task_dev_df.fillna('none', inplace=False)
    all_train_df = task_dev_df

    all_train_df = all_train_df[all_train_df.label != 'none']
    all_train_df["tweet"] = all_train_df["tweet"].apply(lambda x: clean_text(x))
    print(all_train_df.describe())

    all_train_df['text_length'] = all_train_df['tweet'].apply(len)
    print(all_train_df.columns)
    print(all_train_df.describe())

    all_train_df = all_train_df.sample(frac=1).reset_index(drop=True)

    task_test_df = pd.read_csv(f"afrisent-semeval-2023/SubtaskC/test/{la}_test_gold_label.tsv", delimiter="\t", encoding='utf-8')
    tmp = pd.read_csv(f"afrisent-semeval-2023/SubtaskC/test/{la}_test_participants.tsv", delimiter="\t", encoding='utf-8')
    task_test_df["ID"] = tmp["ID"]

    task_test_df["tweet"] = task_test_df["tweet"].apply(lambda x: clean_text(x, clean_flag=True) if la in ["or"] else clean_text(x))
    label_tag = 'label'

    temp = all_train_df.groupby(label_tag).count()['tweet'].reset_index().sort_values(by='tweet', ascending=False)
    label_frequency = {}

    for item in range(len(temp)):
        label_frequency[temp[label_tag][item]] = temp["tweet"][item]

    print("label_tag:", label_tag)
    print("label_frequency: ", label_frequency)

    return all_train_df, task_test_df, label_frequency, label_tag


class SemEval2023_Task12_1_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, df, tokenizer, label2id=None, label_tag=None, emo_dict=None):
        self.args = args
        self.df = df
        self.label_tag = label_tag
        self.labeled = label_tag in df
        self.tokenizer = tokenizer
        self.max_length = args.max_sequence_length
        self.label2id = label2id
        self.emo_dict = emo_dict

    def __getitem__(self, index):
        # ID	tweet	label
        row = self.df.iloc[index]
        if self.args.dynamic_padding == True:
            encoded_output = self.get_input_data_D(row)  # not use
        else:
            encoded_output = self.get_input_data(row)

        if self.labeled:
            label = self.get_label(row)
            encoded_output["label"] = label

        encoded_output["id"] = row["ID"]

        return encoded_output

    def __len__(self):
        return len(self.df)

    def get_input_data_D(self, data):

        # input = data["country_code"] + " </s>" + " </s> " + data["keyword"] + " </s>" + " </s> " + data["text"]
        input = data["tweet"]
        encoded_output = self.tokenizer.encode_plus(
            input,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
        )

        curr_sent = {}
        curr_sent["input_ids"] = encoded_output["input_ids"]
        curr_sent["token_type_ids"] = encoded_output["token_type_ids"]
        curr_sent["attention_mask"] = encoded_output["attention_mask"]

        return curr_sent

    def get_input_data(self, data):
        input = data["tweet"]

        encoded_output = self.tokenizer.encode_plus(
            input,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation_strategy="longest_first",
        )

        curr_sent = {}
        curr_sent["input_ids"] = torch.tensor(encoded_output["input_ids"], dtype=torch.long)
        curr_sent["token_type_ids"] = torch.tensor(encoded_output["token_type_ids"], dtype=torch.long)
        curr_sent["attention_mask"] = torch.tensor(encoded_output["attention_mask"], dtype=torch.long)
        return curr_sent

    def get_label(self, row):
        label_id = self.label2id[row[self.label_tag]]
        assert isinstance(label_id, int) == True
        label = torch.tensor(int(label_id))
        return label


def get_train_val_dataloader(args, train_df, train_idx, val_idx, tokenizer, label_frequency, label2id, label_tag):
    train_data = train_df.iloc[train_idx]
    valid_data = train_df.iloc[val_idx]

    print("start")
    print("train_dataset: ", len(train_data))
    print("valid_dataset: ", len(valid_data))

    train_dataset = SemEval2023_Task12_1_Dataset(args, train_data, tokenizer, label2id, label_tag)
    valid_dataset = SemEval2023_Task12_1_Dataset(args, valid_data, tokenizer, label2id, label_tag)

    sample_weights = []
    for item in train_data[label_tag]:
        weight = label_frequency[item]
        samp_weight = math.sqrt(
            1 / weight
        )
        sample_weights.append(samp_weight)

    if args.use_weighted_sampler:
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        valid_sampler = SequentialSampler(valid_dataset)

    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader, sample_weights


def get_test_dataloader(args, test_df, tokenizer, label2id, label_tag):
    test_dataset = SemEval2023_Task12_1_Dataset(args, test_df, tokenizer, label2id, label_tag)
    test_sampler = SequentialSampler(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
    )

    dataloaders_dict = {"test": test_loader}
    return dataloaders_dict


def save_train_predicts(args, data, label2id):
    data.to_csv(args.train_result_path, index=False)
    y_true = data.iloc[:, 0]
    y_pred = data.iloc[:, 1]
    target_names = list(label2id.keys())  # ['class 0', 'class 1']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=5))


def test_model(args, model, test_dataloader, train_data=False):
    bin_files = [f for f in os.listdir(args.model_save_path) if f.endswith('.bin')]

    prediction_sum = []
    for fold in bin_files:
        model.cuda()
        output_dir = os.path.join(args.model_save_path, fold)
        model.load_state_dict(torch.load(output_dir))
        model.eval()
        prediction = []
        label_s = []
        predit_sum = []
        for batch in test_dataloader["test"]:
            model.eval()
            with torch.no_grad():
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                token_type_ids = batch['token_type_ids'].cuda()
                label = batch['label'].cuda().cpu().detach().numpy()

                output = model(input_ids, attention_mask, token_type_ids, labels=None)
                _, logits = output[0], output[1]

                out_sort = torch.argmax(logits, dim=1).cpu().detach().numpy()
                predit_sum = np.hstack((predit_sum, out_sort))

                logits = F.softmax(logits, dim=1).cpu().detach().numpy()

                if prediction == []:
                    prediction = logits
                    label_s = label
                else:
                    prediction = np.vstack((prediction, logits))
                    label_s = np.hstack((label_s, label))
        if prediction_sum == []:
            prediction_sum = prediction
        else:
            prediction_sum += prediction

        label_sum = label_s
        f1 = f1_score(label_sum, predit_sum, average='weighted')
        acc = accuracy_score(label_sum, predit_sum)
        p = precision_score(label_sum, predit_sum, average='weighted')
        r = recall_score(label_sum, predit_sum, average='weighted')

        print('test_w-f1: {: >4.5f}'.format(f1))
        print('test_acc: {: >4.5f}'.format(acc))
        print('test_p: {: >4.5f}'.format(p))
        print('test_r: {: >4.5f}'.format(r))

    logits = prediction_sum / len(bin_files)
    pd.DataFrame(logits).to_csv(args.test_result_path)
    test_df = np.argmax(logits, axis=1)
    pd.DataFrame(test_df).to_csv(args.submission_path if not train_data else args.submission_path + ".train", sep=',', index=False, header=None)


def main(args):
    file_handler = logging.handlers.TimedRotatingFileHandler(args.log_path, 'D', 1, 7)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    init_logger()
    x = import_module('model.' + args.model_head)
    function_utils.seed_everything(args.seed)
    print(f"Apex AMP Installed : {APEX_INSTALLED}")

    logger.info("*********************** Args Statemnet ******************************")
    logger.info('Args: {}'.format(args))

    train_df, test_df, label_frequency, label_tag = read_dataset(la=args.la)
    label2id = dict((v, k) for k, v in enumerate(label_frequency.keys()))

    label_counts = list(label_frequency.values())
    all_counts = sum(label_counts)
    class_weights = torch.FloatTensor([all_counts / e for e in label_counts]).cuda()
    print("label2id: ", label2id)
    print("class_weights: ", class_weights)

    print("train_dataset: ", len(train_df))
    print("test_dataset: ", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<e>", "<e/>"]})

    model = x.ClassifierModel(args, class_weights, tokenizer).to(args.device)
    test_dataloader = get_test_dataloader(args, test_df, tokenizer, label2id, label_tag)

    test_model(args, model, test_dataloader)

    print("finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SemEval2023 Task12 Code')
    parser.add_argument('--model_task', type=str, default='muticlassification')

    parser.add_argument('--la', type=str, default='multilingual', help='language: am, dz, ha ig kr ma pcm pt sw ts twi yo')

    # Pretrain Model Select
    parser.add_argument('--model_name', type=str, default='roberta-large',
                        help='MAIN MODEL_TYPE:'
                             "(1) bert-large"
                             "(2) roberta-large"
                             "(3) ernie2-large"
                             "(4) xlm-roberta-large")

    # Support 6 Model Head Structure
    parser.add_argument('--model_head', type=str, default='CLS_POOLING_F',
                        help='MAIN HEAD_POOLING FUNCTION:'
                             "(1) CLS_Pooling_F"
                             "(2) DynamicRouting_Pooling_F"
                             "(3) LSTM_Plus_Attention_Pooling_F"
                             "(4) MAX_AVG_Pooling_F"
                             "(5) All_Pooling_Cat"
                             "(6) Weightedlayer_Pooling_A"
                             "(7) WKPooling_head"
                             "(8) Concat_Pooling_A"
                             "(9) Attention_Pooling_A")

    # Support 2 Different Ways of Muti-Exit
    parser.add_argument("--muti_exit", type=str, default=None,
                        help='MAIN MUTI-EXIT FUNCTION: (1) Muti_exit with Different FC Layers'
                             '(2) Muti_exit with Same FC Layers')
    # Doing Further Pretraining
    parser.add_argument("--pretraining_PLM_Model", type=bool, default=True,
                        help='USING COMPETITION DATA FURTHER PRETRAING PLM MODEL')

    # ----------------------------------------------------------------------------------------------------------------------
    # Data and Result Path
    parser.add_argument("--train_df", default="data/task4_1_train.txt", type=str,
                        help="Training Data File Location: Task4_1: data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv")
    parser.add_argument("--test_df", default="data/task4_1_test.txt", type=str, help="Test data file location:")
    parser.add_argument("--additional_df", default=None, type=str, help="Additional data location ")
    parser.add_argument("--country_code", default="data/country code.txt", type=str, help="Country Code Data Location")
    parser.add_argument("--train_result_path", default="output/train_predict.csv", type=str)
    parser.add_argument("--test_result_path", default="output/test_logits.csv", type=str)
    parser.add_argument("--model_save_path", default="output/", type=str, help="Model Saving Location")
    parser.add_argument("--log_path", default="logs/logs.log", type=str)
    parser.add_argument("--submission_path", default="output/task1.txt", type=str)
    parser.add_argument("--result_dic_path", default="output/", type=str)

    # ----------------------------------------------------------------------------------------------------------------------
    # Apx and gradient accumulation
    parser.add_argument("--fp16", default=False, type=str, help="")
    parser.add_argument("--fp16_opt_level", default="01", type=str, help="")
    parser.add_argument('--gradient_accumulation_steps', type=float, default=1.0, help="Gradient Accumulation")

    # ----------------------------------------------------------------------------------------------------------------------
    # Optimizer and scheduler
    parser.add_argument("--optimizer_type", default='AdamW', type=str,
                        help="Main Optimizer Type: (1) Adam(2) AdamW(3) LAMB(4) MADGRAD")
    parser.add_argument("--higher_optimizer", default="lookahead", type=str, help="((1) lookahead, (2)swa, (3) None")
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--epsilon", default=1e-8, type=float, help="")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--decay_name", default='cosine_warmup', type=str, help="")
    parser.add_argument("--warmup_ratio", default=0.1, type=float)

    parser.add_argument('--not_use_LLRD_flag', action='store_true', default=False, help='not use LLRD flag')

    # ----------------------------------------------------------------------------------------------------------------------
    # Model Hyperparameter
    parser.add_argument("--batch_interval", default=5, type=float, help="Batch Interval for doing Evaluation")
    parser.add_argument('--logging_steps', type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--patience", default=15, type=int, help="Early-Stopping Batch Interval")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--dynamic_padding", default=False, type=int, help="(1) True;(2) False")
    parser.add_argument('--max_sequence_length', type=int, default=250,
                        help="Choose Max Length, If Dynamic Dadding is False")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")  # 6
    parser.add_argument("--layer_start", default=21, type=int, help="weightedlayer")
    parser.add_argument("--layer_weights", default=None, type=str, help="weightedlayer")
    parser.add_argument("--gllrd_rate", default=1.0, type=float)
    parser.add_argument("--head_rate", default=2, type=float)
    # -----------------------------------------------------------------------------------------------------------------------
    # for multi-sample dropout
    parser.add_argument("--mutisample_dropout", default=True, type=str, help="multi-sample dropout rate")
    parser.add_argument("--dropout_rate", default=0.4, type=float, help="multi-sample dropout rate")
    parser.add_argument("--dropout_num", default=1, type=int, help="how many dropout samples to draw")
    parser.add_argument("--dropout_action", default="sum", type=str, help="sum, avg")

    # -----------------------------------------------------------------------------------------------------------------------
    # Deal with Unbalanced Sample Distribution
    parser.add_argument('--use_weighted_sampler', action='store_true', default=False, help='')
    parser.add_argument('--use_class_weights', action='store_true', default=False, help='')
    # ----------------------------------------------------------------------------------------------------------------------
    # Loss Function
    parser.add_argument("--loss_fct_name", type=str, default="CrossEntropy",
                        help="(1) CrossEntropy loss; (2) Focal loss; (3) Dice loss")
    parser.add_argument("--focal_loss_gamma", default=0.0, type=float, help="gamma in focal loss")

    parser.add_argument('--contrastive_loss_flag', action='store_true', default=False, help='use contrastive_loss')
    parser.add_argument("--contrastive_loss", default="SupCon", type=str, help="(1) NTXent loss (2) SupCon loss")
    parser.add_argument("--what_to_contrast", default="sample", type=str,
                        help="(1) sample; (2) sample_with_class_embeddings")
    parser.add_argument("--contrastive_loss_weight", default=0.0, type=float, help="loss weight for ntxent")
    parser.add_argument("--contrastive_loss_weight2", default=0.0, type=float, help="loss weight for ntxent")
    parser.add_argument("--contrastive_temperature", default=0.1, type=float, help="temperature for contrastive loss")

    # ----------------------------------------------------------------------------------------------------------------------
    # Adversarial Training
    parser.add_argument('--adversary_flag', action='store_true', default=False, help='use adversarial_method')
    parser.add_argument("--at_method", default='fgm', type=str, help="(1) fgm; (2) pgd; (3) None")
    parser.add_argument("--at_rate", default=0.0, type=float, help="for a fraction of iterations, do at training;")
    parser.add_argument("--emb_names", default="word_embedding, encoder.layer.0", type=str,
                        help="(1) word_embedding; (2) encoder.layer.0")
    parser.add_argument("--fgm_epsilon_for_at", default=1, type=float,
                        help="fgm_epsilon coefficient for adv training: step size")
    parser.add_argument("--pgd_epsilon_for_at", default=0.5, type=float,
                        help="epsilon coefficient for adv training: step size")
    parser.add_argument("--alpha_for_at", default=0.1, type=float,
                        help="alpha coefficient for adv training: step size for PGD")
    parser.add_argument("--steps_for_at", default=3, type=float, help="num of steps at each adv sample: for PGD")

    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--pretrain_model_path", default='Davlan/afro-xlmr-large', type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--device", default="cuda", type=str, help="device")

    args = parser.parse_args()
    if not os.path.exists('logs'): os.makedirs('logs')

    main(args)
