import os

import argparse
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPRobertaCometDataset
from model import DialogueCRN
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
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


def get_IEMOCAP_bert_loaders(path=None, batch_size=32, num_workers=0, pin_memory=False, valid_rate=0.1):
    trainset = IEMOCAPRobertaCometDataset(path=path, split='train-valid')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPRobertaCometDataset(path=path, split='test')
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, scheduler=None, cuda_flag=False, feature_type='text',
                        target_names=None, tensorboard=False, contrast_hidden_flag=False, contrast_weight=0.0, contrast_weight2=0.0, adversary_flag=False,
                        adv_trainer=None, at_method=None, at_rate=0.0, loss_f2=None, eval_cluster_flag=False, gradient_accumulation_steps=16):
    assert not train_flag or optimizer != None
    losses, preds, labels = [], [], []
    preds2 = []

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
        loss = loss_f(log_prob, label)

        if contrast_weight > 0:
            cl_loss = loss_f2(log_prob2 if contrast_hidden_flag else log_prob, label)
            print("w: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight, loss.item(), len(label) * cl_loss.item()))
            loss += cl_loss * contrast_weight * len(label)

        if gradient_accumulation_steps > 0: loss = loss / gradient_accumulation_steps

        if eval_cluster_flag: preds2.append(log_prob.cpu().detach().numpy())
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()  # retain_graph=True

            if adversary_flag:
                at_flag = random.uniform(0, 1) <= at_rate

                if at_flag:
                    print("# at_flag: ", at_flag)
                    if at_method == "fgm":
                        adv_trainer.backup_grad()
                        adv_trainer.attack()
                        model.zero_grad()
                        outputs_at, log_prob2_at = model(r1, qmask, seq_lengths)
                        loss_at = loss_f(outputs_at, label)

                        if contrast_weight2 > 0:
                            cl_loss_at = loss_f2(log_prob2_at if contrast_hidden_flag else outputs_at, label)
                            print("at: w2: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight2, loss_at.item(), len(label) * cl_loss_at.item()))
                            loss_at += cl_loss_at * contrast_weight2 * len(label)

                        if gradient_accumulation_steps > 0: loss_at = loss_at / gradient_accumulation_steps

                        loss_at.backward()
                        adv_trainer.restore_grad()
                        adv_trainer.restore()
                    elif at_method == "pgd":
                        adv_trainer.backup_grad()
                        steps_for_at = 3
                        for t in range(steps_for_at):
                            adv_trainer.attack(is_first_attack=(t == 0))
                            model.zero_grad()
                            outputs_at, log_prob2_at = model(r1, qmask, seq_lengths)
                            loss_at = loss_f(outputs_at, label)

                            if contrast_weight2 > 0:
                                cl_loss_at = loss_f2(log_prob2_at if contrast_hidden_flag else outputs_at, label)
                                print("at: w2: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight2, loss_at.item(), len(label) * cl_loss_at.item()))
                                loss_at += cl_loss_at * contrast_weight2 * len(label)
                            if gradient_accumulation_steps > 0: loss_at = loss_at / gradient_accumulation_steps

                            loss_at.backward()
                        adv_trainer.restore_grad()
                        adv_trainer.restore()

            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            if grad_acc_flag:
                optimizer.step()
                optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(metrics.classification_report(labels, preds, target_names=target_names, digits=4))
    all_matrix.append(["ACC"])
    for i in range(len(target_names)):
        all_matrix[-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    if eval_cluster_flag:
        preds2 = np.array(np.concatenate(preds2))

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds, preds2]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--status', type=str, default='test', help='optional status: train/test/test_attack')

    parser.add_argument('--feature_type', type=str, default='text', help='feature type multi/text/acouf')

    parser.add_argument('--data_dir', type=str, default='../data/iemocap/iemocap_features_roberta.pkl', help='dataset dir: IEMOCAP_features.pkl')

    parser.add_argument('--output_dir', type=str, default='../outputs/iemocap/sacl_lstm_iemocap', help='saved model dir')

    parser.add_argument('--load_model_state_dir', type=str, default='../sacl_lstm_best_models/sacl_lstm_iemocap/4/loss_sacl-lstm.pkl', help='load model state dir')

    parser.add_argument('--base_model', default='LSTM', help='base model, LSTM/GRU/Linear')

    parser.add_argument('--base_layer', type=int, default=2, help='the number of base model layers,1/2')

    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=20, help='early stop')

    parser.add_argument('--batch_size', type=int, default=2, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate, 0.0/0.1')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--step_s', type=int, default=0, help='the number of reason turns at situation-level')

    parser.add_argument('--step_p', type=int, default=0, help='the number of reason turns at speaker-level')

    parser.add_argument('--gamma', type=float, default=0, help='gamma 0/0.5/1/2/5')

    parser.add_argument('--scl_hidden_flag', action='store_true', default=False, help='put scl loss on hidden or output')

    parser.add_argument('--scl_t', type=float, default=0.1, help='contrastive_temperature 0.07/0.1/0.5/1')

    parser.add_argument('--scl_w', type=float, default=0.05, help='contrastive_weight 0.01/0.1/1')

    parser.add_argument('--scl_w2', type=float, default=0.5, help='contrastive_weight 0.01/0.1/1')

    parser.add_argument('--at_rate', type=float, default=1.0, help='at_rate 0.1/0.5/1')

    parser.add_argument('--at_epsilon', type=float, default=5, help='at_epsilon 1/5')

    parser.add_argument('--adversary_flag', action='store_true', default=False, help='does not use adversary training')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')

    parser.add_argument('--class_weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='enables tensorboard log')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='gradient_accumulation_steps')

    args = parser.parse_args()
    print(args)

    epochs, batch_size, status, output_path, data_path, base_model, base_layer, feature_type = \
        args.epochs, args.batch_size, args.status, args.output_dir, args.data_dir, args.base_model, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    reason_steps = [args.step_s, args.step_p]

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    # IEMOCAP dataset
    n_classes, n_speakers, hidden_size, input_size = 6, 2, 128, None
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    class_weights = torch.FloatTensor([1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555])
    if feature_type == 'multi':
        input_size = 712
    elif feature_type in ['text', 'acouf']:
        input_size = 1024  # 100
    else:
        print('Error: feature_type not set.')
        exit(0)

    seed_everything(seed=args.seed)
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
        loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None, size_average=False)

        if contrast_weight > 0 or contrast_weight2 > 0:
            from pytorch_metric_learning.distances import DotProductSimilarity
            from pytorch_metric_learning.losses import NTXentLoss, SupConLoss

            loss_f2 = NTXentLoss(temperature=contrast_temperature, distance=DotProductSimilarity())
    else:
        # NLLLoss
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None)

    train_loader, valid_loader, test_loader = get_IEMOCAP_bert_loaders(path=data_path, batch_size=batch_size, num_workers=0, valid_rate=args.valid_rate)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = None
    # contrast_criterion = None
    contrastive_loss = "NTXentLoss"

    adversary_flag = args.adversary_flag
    at_method = 'fgm'
    at_rate = args.at_rate
    emb_names = [
        'rnn.weight_ih_l1', 'rnn.bias_ih_l1',
        "rnn.weight_ih_l1_reverse", 'rnn.bias_ih_l1_reverse',
        "rnn_parties.weight_ih_l1", 'rnn_parties.bias_ih_l1',
        "rnn_parties.weight_ih_l1_reverse", "rnn_parties.bias_ih_l1_reverse"
    ]
    adv_trainer = None
    if adversary_flag:
        from at_training import FGM, PGD

        if at_method == "fgm":
            adv_trainer = FGM(
                model,
                epsilon=args.at_epsilon,
                emb_names=emb_names,
            )
        elif at_method == "pgd":
            adv_trainer = PGD(
                model,
                epsilon=args.at_epsilon,
                alpha=0.1,  # at_alpha
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

            train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=train_loader, epoch=e, train_flag=True,
                                                                            optimizer=optimizer, scheduler=scheduler, cuda_flag=cuda_flag,
                                                                            feature_type=feature_type, target_names=target_names,
                                                                            contrast_hidden_flag=contrast_hidden_flag, contrast_weight=contrast_weight,
                                                                            contrast_weight2=contrast_weight2, adversary_flag=adversary_flag,
                                                                            adv_trainer=adv_trainer, at_method=at_method, at_rate=at_rate, loss_f2=loss_f2,
                                                                            gradient_accumulation_steps=args.gradient_accumulation_steps)
            valid_loss, valid_acc, valid_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader, epoch=e, cuda_flag=cuda_flag,
                                                                            feature_type=feature_type, target_names=target_names, loss_f2=loss_f2,
                                                                            gradient_accumulation_steps=args.gradient_accumulation_steps)
            test_loss, test_acc, test_fscore, test_metrics, label_pred = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader, epoch=e,
                                                                                             cuda_flag=cuda_flag, feature_type=feature_type,
                                                                                             target_names=target_names, loss_f2=loss_f2,
                                                                                             gradient_accumulation_steps=args.gradient_accumulation_steps)
            all_test_fscore.append(test_fscore)
            all_test_acc.append(test_acc)

            if args.valid_rate > 0:
                eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
            else:
                eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
            if e == 0 or best_eval_fscore < eval_fscore:
                patience = 0
                best_epoch, best_eval_fscore = e, eval_fscore
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
            print(confusion_matrix(y_true=label_pred[0], y_pred=label_pred[1], normalize="true"))

            if patience >= args.patience and patience2 >= args.patience:
                print('Early stoping...', patience, patience2)
                break

        print('Final Test performance...')
        print('Early stoping...', patience, patience2)
        # print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore,
        #                                                                                             all_test_acc[best_epoch] if best_epoch >= 0 else 0,
        #                                                                                             all_test_fscore[best_epoch] if best_epoch >= 0 else 0))
        print('Eval-metric: Loss, Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch2,
                                                                                all_test_acc[best_epoch2] if best_epoch2 >= 0 else 0,
                                                                                all_test_fscore[best_epoch2] if best_epoch2 >= 0 else 0))

    elif status == 'test':
        start_time = time.time()
        model.load_state_dict(torch.load(args.load_model_state_dir))
        test_loss, test_acc, test_fscore, test_metrics, label_pred = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader,
                                                                                         cuda_flag=cuda_flag, feature_type=feature_type,
                                                                                         target_names=target_names, loss_f2=loss_f2, eval_cluster_flag=True,
                                                                                         gradient_accumulation_steps=args.gradient_accumulation_steps)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, 0)
        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(test_metrics[0])
        print(test_metrics[1])

        print("confusion_matrix: ")
        print(confusion_matrix(y_true=label_pred[0], y_pred=label_pred[1], normalize="true"))

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score, silhouette_score, \
            calinski_harabasz_score, davies_bouldin_score
        from sklearn.cluster import KMeans

        y, y_p, X = label_pred[0], label_pred[1], label_pred[2]
        ari = adjusted_rand_score(y, y_p)
        nmi = normalized_mutual_info_score(y, y_p)
        ami = adjusted_mutual_info_score(y, y_p)
        fmi = fowlkes_mallows_score(y, y_p)
        print("[Supervised Metrics with label y] ARI: {:.4f}, NMI: {:.4f}, AMI: {:.4f}, FMI: {:.4f}".format(ari, nmi, ami, fmi))

        print("using kmeans ....")
        kmeans = KMeans(n_clusters=n_classes)
        y_pred = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, y_pred)
        ch = calinski_harabasz_score(X, y_pred)
        db = davies_bouldin_score(X, y_pred)
        print("[Unsupervised Metrics with kmeans] Silhouette Coefficient: {:.4f}, Calinski-Harabasz Index: {:.4f}, Davies-Bouldin Index: {:.4f}".format(
            silhouette, ch, db))


    else:
        print('the status must be one of train/test')
        exit(0)
