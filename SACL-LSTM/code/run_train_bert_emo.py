import os

import argparse
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import EmoryNLPRobertaCometDataset
from model import DialogueCRN
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score , confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from loss import FocalLoss


def seed_everything(seed=2021):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    np.random.shuffle(idx)  # shuffle for training data
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_EmoryNLP_bert_loaders(path, batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = EmoryNLPRobertaCometDataset(path, 'train', classify)
    validset = EmoryNLPRobertaCometDataset(path, 'valid', classify)
    testset = EmoryNLPRobertaCometDataset(path, 'test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True
                              )

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text', target_names=None,
                        tensorboard=False, contrast_hidden_flag=False, contrast_weight=0.0, contrast_weight2=0.0,
                        adversary_flag=False, adv_trainer=None, at_method=None, at_rate=0.0, situ_rate=1.0, speaker_rate=0.0, at_pgd_step=3,loss_f2=None, gradient_accumulation_steps=2):
    assert not train_flag or optimizer != None
    losses, preds, labels, masks = [], [], [], []

    if train_flag:
        model.train()
    else:
        model.eval()

    if train_flag: optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        grad_acc_flag = step > 0 and ((step % gradient_accumulation_steps == 0) or step == len(dataloader) - 1)


        r1, qmask, umask, label2 = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, log_prob2 = model(r1, qmask, seq_lengths)
        label = torch.cat([label2[j][:seq_lengths[j]] for j in range(len(label2))])
        umask2 = torch.cat([umask[j][:seq_lengths[j]] for j in range(len(umask))])
        loss = loss_f(log_prob, label)

        if contrast_weight>0:
            cl_loss = loss_f2(log_prob2 if contrast_hidden_flag else log_prob, label)
            print("w: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight, loss.item(), len(label) * cl_loss.item()))
            loss += cl_loss * len(label) * contrast_weight

        preds.append(torch.argmax(log_prob, 1).data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        masks.append(umask2.view(-1).cpu().numpy())

        losses.append(loss.item())

        if train_flag:
            loss.backward()

            if adversary_flag:
                rand_rate = random.uniform(0, 1)
                at_flag =  rand_rate<= at_rate

                if at_flag:
                    print("# at_flag: ", at_flag)
                    if rand_rate <= at_rate * situ_rate:
                        # situ 1.0
                        adv_trainer_l = adv_trainer[0]
                    elif rand_rate <= at_rate * (situ_rate + speaker_rate):
                        adv_trainer_l = adv_trainer[1]
                    else:
                        adv_trainer_l = adv_trainer[2]

                    print("# at_flag: {}, rand_rate: {}, situ_rate:{}, speaker_rate:{}".format(at_flag, rand_rate, situ_rate, speaker_rate))


                    if at_method == "fgm":
                        adv_trainer_l.backup_grad()
                        adv_trainer_l.attack()
                        model.zero_grad()
                        outputs_at, log_prob2_at = model(r1, qmask, seq_lengths)
                        loss_at = loss_f(outputs_at, label)

                        if contrast_weight2>0:
                            cl_loss_at = loss_f2(log_prob2_at if contrast_hidden_flag else outputs_at, label)
                            print("at: w2: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight2, loss_at.item(), len(label) * cl_loss_at.item()))
                            loss_at += cl_loss_at * len(label) * contrast_weight2


                        loss_at.backward()
                        adv_trainer_l.restore_grad()
                        adv_trainer_l.restore()
                    elif at_method == "pgd":
                        adv_trainer.backup_grad()
                        for t in range(at_pgd_step):
                            adv_trainer.attack(is_first_attack=(t == 0))
                            model.zero_grad()
                            outputs_at, log_prob2_at = model(r1, qmask, seq_lengths)
                            loss_at = loss_f(outputs_at, label)

                            if contrast_weight2 > 0:
                                cl_loss_at = loss_f2(log_prob2_at if contrast_hidden_flag else outputs_at, label)
                                print("at: w2: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight2, loss_at.item(), len(label) * cl_loss_at.item()))
                                loss_at += cl_loss_at * len(label) * contrast_weight2

                            loss_at.backward()
                        adv_trainer.restore_grad()
                        adv_trainer.restore()

            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            if grad_acc_flag:
                optimizer.step()
                optimizer.zero_grad()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(
        metrics.classification_report(labels, preds, target_names=target_names if target_names else None, sample_weight=masks, digits=4))
    all_matrix.append(["ACC"])
    for i in range(len(target_names)):
        all_matrix[-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--status', type=str, default='test', help='optional status: train/test')

    parser.add_argument('--feature_type', type=str, default='text', help='feature type, multi/text/audio')

    parser.add_argument('--data_dir', type=str, default='../data/emorynlp/emorynlp_features_roberta.pkl', help='dataset dir')

    parser.add_argument('--output_dir', type=str, default='../outputs/emorynlp/sacl_lstm_emorynlp', help='saved model dir')

    parser.add_argument('--load_model_state_dir', type=str, default='../sacl_lstm_best_models/sacl_lstm_emorynlp/1/f1_sacl-lstm.pkl', help='load model state dir')

    parser.add_argument('--base_model', default='LSTM', help='base model, LSTM/GRU/Linear')

    parser.add_argument('--base_layer', type=int, default=1, help='the number of base model layers, 1/2')

    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=20, help='early stop')

    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size 32')

    parser.add_argument('--use_valid_flag', action='store_true', default=False, help='use valid set')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--step_s', type=int, default=0, help='the number of reason turns at situation-level,3')

    parser.add_argument('--step_p', type=int, default=0, help='the number of reason turns at speaker-level,0')

    parser.add_argument('--gamma', type=float, default=1, help='gamma 0/0.5/1/2')

    parser.add_argument('--scl_hidden_flag', action='store_true', default=False, help='put scl loss on hidden or output')

    parser.add_argument('--scl_t', type=float, default=0.1, help='contrastive_temperature 0.07/0.1/0.5/1')

    parser.add_argument('--scl_w', type=float, default=0.1, help='contrastive_weight 0.01/0.1/1')

    parser.add_argument('--scl_w2', type=float, default=0.0, help='contrastive_weight 0.01/0.1/1')

    parser.add_argument('--at_rate', type=float, default=0.5, help='at_rate 0.1/0.5/1')

    parser.add_argument('--situ_rate', type=float, default=1.0, help='at_rate 0.1/0.5/1')

    parser.add_argument('--speaker_rate', type=float, default=0.0, help='at_rate 0.1/0.5/1')

    parser.add_argument('--at_epsilon', type=float, default=1.0, help='at_epsilon 1/5')

    parser.add_argument('--adversary_flag', action='store_true', default=False, help='does not use adversary training')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')

    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--cls_type', type=str, default='emotion', help='choose between sentiment or emotion')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='gradient_accumulation_steps')

    args = parser.parse_args()
    print(args)

    epochs, batch_size, status, output_path, data_path, load_model_state_dir, base_model, base_layer, feature_type = \
        args.epochs, args.batch_size, args.status, args.output_dir, args.data_dir, args.load_model_state_dir, args.base_model, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    reason_steps = [args.step_s, args.step_p]

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    # MELD dataset
    n_speakers, hidden_size, input_size = 9, 128, None  # 128
    if args.cls_type.strip().lower() == 'emotion':
        n_classes = 7
        # {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        target_names = ['joy', 'mad', 'pea', 'neu', 'sad', 'pow', 'sca']

        class_weights = torch.FloatTensor(
            [1 / 0.221281599, 1 / 0.103703704, 1 / 0.084656085, 1 / 0.330041152, 1 / 0.061728395, 1 / 0.073015873, 1 / 0.125573192])
        class_weights = torch.log(class_weights)
        # class_weights = torch.pow(class_weights, 0.5)

    else:
        # sentiment
        n_classes = 3
        target_names = ['0', '1', '2']
        class_weights = torch.FloatTensor([1.0, 1.0, 1.0])
    if feature_type == 'multi':
        input_size = 900
    elif feature_type == 'text':
        input_size = 1024  # 600
    elif feature_type == 'audio':
        input_size = 300
    else:
        print('Error: feature_type not set.')
        exit(0)

    seed_everything(seed=args.seed)

    train_loader, valid_loader, test_loader = get_EmoryNLP_bert_loaders(data_path,
                                                                        batch_size=batch_size,
                                                                        classify=args.cls_type,
                                                                        num_workers=0)

    model = DialogueCRN(base_model=base_model,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        dropout=args.dropout,
                        cuda_flag=cuda_flag,
                        reason_steps=reason_steps)

    if cuda_flag:
        print('Running on GPU')
        # torch.cuda.set_device(1)
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    name = 'SACL-LSTM'
    print('{} with {} as base model.'.format(name, base_model))
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))
    for n, p in model.named_parameters():
        print(n, p.size(), p.requires_grad)

    contrast_hidden_flag, contrast_temperature, contrast_weight, contrast_weight2 = args.scl_hidden_flag, args.scl_t, args.scl_w, args.scl_w2
    loss_f2 = None
    if args.loss == 'FocalLoss':
        # FocalLoss
        loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None, size_average=False)  # default True

        if contrast_weight>0 or contrast_weight2>0:
            from pytorch_metric_learning.distances import DotProductSimilarity
            from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
            loss_f2 = NTXentLoss(temperature=contrast_temperature, distance=DotProductSimilarity())

    else:
        # NLLLoss
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    adversary_flag = args.adversary_flag
    at_method = 'fgm'
    at_rate = args.at_rate
    emb_names = [
        'rnn.weight_ih_l0', 'rnn.bias_ih_l0',
        "rnn.weight_ih_l0_reverse", 'rnn.bias_ih_l0_reverse',
    ]

    emb_names2=[
    "rnn_parties.weight_ih_l0", 'rnn_parties.bias_ih_l0',
    "rnn_parties.weight_ih_l0_reverse", 'rnn_parties.bias_ih_l0_reverse',
    ]

    emb_names3 = [
        'rnn.weight_ih_l0', 'rnn.bias_ih_l0',
        "rnn.weight_ih_l0_reverse", 'rnn.bias_ih_l0_reverse',
        "rnn_parties.weight_ih_l0", 'rnn_parties.bias_ih_l0',
        "rnn_parties.weight_ih_l0_reverse", 'rnn_parties.bias_ih_l0_reverse',
    ]

    adv_trainer = None

    if adversary_flag:
        from at_training import FGM, PGD

        if at_method == "fgm":
            adv_trainer1 = FGM(
                model,
                epsilon=args.at_epsilon,
                emb_names=emb_names,
            )
            adv_trainer2 = FGM(
                model,
                epsilon=args.at_epsilon,
                emb_names=emb_names2,
                )

            adv_trainer3 = FGM(
                model,
                epsilon=args.at_epsilon,
                emb_names=emb_names3,
            )
            adv_trainer = [adv_trainer1, adv_trainer2, adv_trainer3]


        elif at_method == "pgd":
            adv_trainer = PGD(
                model,
                epsilon=args.at_epsilon,
                alpha=0.1,  # args.at_alpha,
                emb_names=emb_names,
            )
        else:
            raise ValueError("un-supported adversarial training method: {} !!!".format(args.at_method))


    output_path = os.path.join(output_path, '{}'.format(args.seed))
    if not os.path.exists(output_path): os.makedirs(output_path)


    if status == 'train':
        all_test_fscore, all_test_acc = [], []
        best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
        patience2 = 0
        for e in range(epochs):
            start_time = time.time()
            train_loss, train_acc, train_fscore, train_metrics, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=train_loader, epoch=e,
                                                                                        train_flag=True,
                                                                                        optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type,
                                                                                        target_names=target_names,
                                                                                        contrast_hidden_flag=contrast_hidden_flag, contrast_weight=contrast_weight, contrast_weight2=contrast_weight2,
                                                                                        adversary_flag=adversary_flag, adv_trainer=adv_trainer,
                                                                                        at_method=at_method, at_rate=at_rate, situ_rate=args.situ_rate,speaker_rate=args.speaker_rate,loss_f2=loss_f2,
                                                                                        gradient_accumulation_steps=args.gradient_accumulation_steps)
            valid_loss, valid_acc, valid_fscore, valid_metrics, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader, epoch=e,
                                                                                        cuda_flag=cuda_flag, feature_type=feature_type,
                                                                                        target_names=target_names, loss_f2=loss_f2,
                                                                                        gradient_accumulation_steps=args.gradient_accumulation_steps)
            test_loss, test_acc, test_fscore, test_metrics, label_pred = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader, epoch=e,
                                                                                    cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names,
                                                                                    loss_f2=loss_f2,gradient_accumulation_steps=args.gradient_accumulation_steps)
            all_test_fscore.append(test_fscore)
            all_test_acc.append(test_acc)

            if args.use_valid_flag:
                eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
            else:
                eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
            if e == 0 or best_eval_fscore < eval_fscore:
                patience = 0
                best_epoch, best_eval_fscore = e, eval_fscore
                if not os.path.exists(output_path): os.makedirs(output_path)
                save_model_dir = os.path.join(output_path, 'f1_{}.pkl'.format(name).lower())
                torch.save(model.state_dict(), save_model_dir)
            else:
                patience += 1
            if best_eval_loss is None:
                best_eval_loss = eval_loss
                best_epoch2 = 0
            else:
                if eval_loss < best_eval_loss:
                    best_epoch2, best_eval_loss = e, eval_loss
                    patience2 = 0
                    if not os.path.exists(output_path): os.makedirs(output_path)
                    # save_model_dir = os.path.join(output_path, 'loss_{}_{}.pkl'.format(name, e).lower())
                    # torch.save(model.state_dict(), save_model_dir)
                    save_model_dir = os.path.join(output_path, 'loss_{}.pkl'.format(name).lower())
                    torch.save(model.state_dict(), save_model_dir)

                else:
                    patience2 += 1

            if args.tensorboard:
                writer.add_scalar('train: accuracy/f1/loss', train_acc / train_fscore / train_loss, e)
                writer.add_scalar('valid: accuracy/f1/loss', valid_acc / valid_fscore / valid_loss, e)
                writer.add_scalar('test: accuracy/f1/loss', test_acc / test_fscore / test_loss, e)
                writer.close()

            print(
                'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                    format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                           round(time.time() - start_time, 2)))
            print(test_metrics[0])
            print(test_metrics[1])

            print("confusion_matrix: ")
            print(confusion_matrix(y_true=label_pred[0], y_pred=label_pred[1], normalize = "true"))


            if patience >= args.patience and patience2 >= args.patience:
                print('Early stoping...', patience, patience2)
                break

        print('Final Test performance...')
        print('Early stoping...', patience, patience2)
        print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore,
                                                                                                    all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                                                                    all_test_fscore[best_epoch] if best_epoch >= 0 else 0))
        print('Eval-metric: Loss, Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch2,
                                                                                all_test_acc[best_epoch2] if best_epoch2 >= 0 else 0,
                                                                                all_test_fscore[best_epoch2] if best_epoch2 >= 0 else 0))


    elif status == 'test':
        start_time = time.time()
        model.load_state_dict(torch.load(args.load_model_state_dir))
        test_loss, test_acc, test_fscore, test_metrics, test_outputs = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader,
                                                                                           cuda_flag=cuda_flag, feature_type=feature_type,
                                                                                           target_names=target_names, gradient_accumulation_steps=args.gradient_accumulation_steps)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, 0)
            writer.close()
        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(test_metrics[0])
        print(test_metrics[1])
    else:
        print('the status must be one of train/test')
        exit(0)
