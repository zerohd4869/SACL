import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from argparse import ArgumentParser

label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
id2label = dict((v, k) for k, v in label2id.items())


def main():
    parser = ArgumentParser()
    parser.add_argument("--la", type=str, default=None, help="tg, or, multilingual")
    args = parser.parse_args()
    la = args.la


    test_file = f"afrisent-semeval-2023/SubtaskC/test/{la}_test_participants.tsv"
    gold_dir = f"afrisent-semeval-2023/SubtaskC/test/{la}_test_gold_label.tsv"


    test_pred_file = f"./outputs/afro-xlmr-large/sacl_xlmr-{la}/task1.txt"

    test_df = pd.read_csv(test_file, delimiter="\t", encoding='utf-8')  # lineterminator='\n'
    test_pred_df = pd.read_csv(test_pred_file, header=None, index_col=False, delimiter="\t", encoding='utf-8')  # header=None, index_col=None

    assert len(test_df) == len(test_pred_df)
    test_df['label'] = test_pred_df[0].apply(lambda id: id2label[id])

    print(test_df.columns)
    out_path = "/".join(test_pred_file.split('/')[:-1])
    out_file = os.path.join(out_path, f"pred_{la}.tsv")
    test_df[['ID', 'label']].to_csv(out_file, sep='\t', index=False)
    print("finished!")

    submission_df = test_df
    gold_df = pd.read_csv(gold_dir, sep='\t')
    assert len(gold_df) == len(test_df)

    f1 = f1_score(y_true=gold_df["label"], y_pred=submission_df["label"], average="weighted")
    recall = recall_score(y_true=gold_df["label"], y_pred=submission_df["label"], average="weighted")
    precision = precision_score(y_true=gold_df["label"], y_pred=submission_df["label"], average="weighted")
    acc = accuracy_score(gold_df["label"], submission_df["label"])

    print('test_w-f1: {: >4.5f}'.format(f1))
    print('test_acc: {: >4.5f}'.format(acc))
    print('test_p: {: >4.5f}'.format(precision))
    print('test_r: {: >4.5f}'.format(recall))

    print("confusion_matrix: ")
    print(confusion_matrix(y_true=gold_df["label"], y_pred=submission_df["label"], normalize="true"))

    print("0220 finished!")
