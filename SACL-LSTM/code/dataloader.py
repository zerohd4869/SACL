import pandas as pd
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        elif split == 'train-valid':
            self.keys = [x for x in self.trainIds] + [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]



class MELDRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
            self.roberta1_avg = None

        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        elif split == 'train-valid':
            self.keys = [x for x in self.trainIds] + [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]


class EmoryNLPRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''

        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainId, self.testId, self.validId \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        self.trainId = sorted(list(self.trainId), reverse=False) # set to list
        self.testId = sorted(list(self.testId), reverse=False)
        self.validId = sorted(list(self.validId), reverse=False)

        sentiment_labels = {}
        for item in self.emotion_labels:
            array = []
            # 0 negative, 1 neutral, 2 positive
            for e in self.emotion_labels[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array


        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]
        elif split == 'train-valid':
            self.keys = [x for x in self.trainId] + [x for x in self.validId]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        elif classify == 'sentiment':
            self.labels = sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid


    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]
