#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import time
import os
import logging
import torch
import logging
import warnings
import numpy as np
from torch import nn
import multiprocessing
from utils.lamb import Lamb
import torch.optim as optim
from madgrad import MADGRAD
from transformers import AdamW
from torch.optim import Adam
import torch.nn.functional as F
from datetime import timedelta
import torch.nn.functional as F
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def seed_everything(seed_value):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return optimal_value


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class AverageMeter(object):
    # Metric Logger
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


################################################lEARNING RATE SCHEDULER#################################################

def get_optimizer_params_l(args, model):
    # Grouped Layer-wise Learning Rate Decay (GLLRD) FOR LARGE MODEL
    gllrd_rate = args.gllrd_rate  # 2.6 1.6
    head_rate = args.head_rate  # 3 2

    no_decay = ['bias', 'gamma', 'beta']
    # 1e-5 / 1.6
    group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
    # 1e-5
    group2 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.', 'layer.12.', 'layer.13.', 'layer.14.', 'layer.15.']
    # 1e-5 * 1.6
    group3 = ['layer.16.', 'layer.17.', 'layer.18.', 'layer.19.', 'layer.20.', 'layer.21.', 'layer.22.', 'layer.23.']
    group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.', 'layer.8.', 'layer.9.',
                 'layer.10.', 'layer.11.', 'layer.12.', 'layer.13.', 'layer.14.', 'layer.15.', 'layer.16.', 'layer.17.', 'layer.18.', 'layer.19.', 'layer.20.',
                 'layer.21.', 'layer.22.', 'layer.23.']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate / gllrd_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate * gllrd_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
         'lr': args.learning_rate / gllrd_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
         'lr': args.learning_rate * gllrd_rate},
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr': args.learning_rate * head_rate, "momentum": 0.99},
    ]
    return optimizer_grouped_parameters


def get_optimizer_params_b(args, model):
    """
    For Base Model
    """
    no_decay = ['bias', 'gamma', 'beta']
    group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
    group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
    group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
    group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': args.weight_decay},
        # group1 learning_rate/2.6
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate / 1.6},
        # group2 learning_rate
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        # group3 learning_rate * 2.6
        {'params': [p for n, p in model.roberta.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate * 1.6},
        # weight_decay
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': 0.0},
        # group1 learning_rate/2.6
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
         'lr': args.learning_rate / 1.6},
        # group2 learning_rate
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        # group3 learning_rate * 2.6
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
         'lr': args.learning_rate * 1.6},
        # non-transformer-Head
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr': args.learning_rate * 5, "momentum": 0.99},
    ]

    return optimizer_grouped_parameters


def make_optimizer(args, model):
    # optimizer
    optimizer_name = args.optimizer_type
    if args.not_use_LLRD_flag == True:
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.roberta.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        ]
    else:

        if args.hidden_size == 768:
            optimizer_grouped_parameters = get_optimizer_params_b(args, model)
        if args.hidden_size == 1024:
            optimizer_grouped_parameters = get_optimizer_params_l(args, model)

    kwargs_1 = {
        'betas': (0.9, 0.98),
        "weight_decay": args.weight_decay,
        'lr': args.learning_rate,
        'eps': args.epsilon,
        'correct_bias': True
    }
    kwargs_2 = {
        "weight_decay": args.weight_decay,
        'lr': args.learning_rate,
        'betas': (0.9, 0.98),
        'eps': args.epsilon,
        'correct_bias': True  # not args.use_bertadam
    }
    kwargs_3 = {
        #   'lr':args.learning_rate, # lr: float = 1e-2
        'weight_decay': args.weight_decay,  # weight_decay: float = 0
        'eps': args.epsilon,  # eps: float = 1e-6
        'momentum': 0.9,  # momentum: float = 0.9
    }

    if optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, **kwargs_1)
        return optimizer

    elif optimizer_name == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, **kwargs_1)
        return optimizer

    elif optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs_2)
        return optimizer

    elif optimizer_name == "MADGRAD":
        # lr: float = 1e-2, momentum: float = 0.9,
        # weight_decay: float = 0, eps: float = 1e-6,
        optimizer = MADGRAD(optimizer_grouped_parameters, **kwargs_3)
        return optimizer

    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer_name))


def make_scheduler(optimizer, args, decay_name='linear', t_max=None, warmup_steps=None):
    if decay_name == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[3, 6, 9],
            gamma=0.1
        )

    elif decay_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.t_max / 2
        )

    elif decay_name == "cosine_warmup":
        # num_warmup_steps：The number of steps for the warmup phase
        # num_training_steps：The total number of training steps.
        # last_epoch (int, optional, defaults to -1) – The index of the last epoch when resuming training.
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )

    elif decay_name == "linear_schedule_with_warmup":
        # num_warmup_steps：The number of steps for the warmup phase
        # num_training_steps：The total number of training steps.
        # last_epoch (int, optional, defaults to -1) – The index of the last epoch when resuming training.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )

    elif decay_name == "cosine_with_hard_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max,
            num_cycles=args.t_max)

    elif decay_name == "polynomial_decay_with_warmup":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max,
            power=2)

    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_name))
    return scheduler


###################################################EarlyStopping######################################################

class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, config, path, verbose=False, delta=0):
        self.patience = config.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


###################################################LabelSmoothLoss######################################################

class LabelSmoothLoss(nn.Module):
    # labelsmothing with crossentropy loss
    def __init__(self, smoothing=0.04):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def loss_fn(out, label):
    loss_fct = LabelSmoothLoss(smoothing=0.04)
    total_loss = loss_fct(out, label)
    return total_loss


###############################################Jaccard String Score#####################################################

def jaccard_from_logits_string(data, start_logits, end_logits):
    # data, predict_start_position_sum, predict_end_position_sum, label_start_position_sum, label_end_position_sum
    n = start_logits.size(0)
    score = 0

    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

    for i in range(n):
        start_idx = np.argmax(start_logits[i])
        end_idx = np.argmax(end_logits[i])
        text = data["text"][i]
        pred = text[start_idx: end_idx]

        score += jaccard(data["selected_text"][i], pred)

    return score


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    try:
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        return 0


################################################POOLING UTILS FUNCTION#################################################

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a models that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ValueError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    output = tensor.masked_fill((1 - mask).byte(), replace_with)
    return output


################################################DIFFIENT POOLING FUNCTION###############################################

# 1 SELF ATTENTION POOLING

class SelfAttnAggregator(nn.Module):
    """
    A ``SelfAttnAggregator`` is a self attn layers.  As a
    :class:`SelfAttnAggregator`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, output_dim,
                 attn_vector=None) -> None:
        super(SelfAttnAggregator, self).__init__()

        self.output_dim = output_dim

        self.attn_vector = None
        if attn_vector:
            self.attn_vector = attn_vector
        else:
            self.attn_vector = nn.Linear(
                self.output_dim,
                1
            )

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_self_attn_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.attn_vector(
            input_tensors
        ).squeeze(2)
        self_weights = masked_softmax(self_attentive_logits, mask)
        input_self_attn_pooled = weighted_sum(input_tensors, self_weights)

        return input_self_attn_pooled


# 2 AVERAGE POOLING

class AvgPoolerAggregator(nn.Module):
    """
    A ``AvgPoolerAggregator`` is a avg pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, ) -> None:
        super(AvgPoolerAggregator, self).__init__()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_max_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """
        # if mask is not None:
        #     # Simple Pooling layers
        #     input_tensors = replace_masked_values(
        #         input_tensors, mask.unsqueeze(2), 0
        #     )

        tokens_avg_pooled = torch.mean(input_tensors, 1)

        return tokens_avg_pooled


# 3 MAXPOOLING

class MaxPoolerAggregator(torch.nn.Module):
    """
    A ``MaxPoolerAggregator`` is a max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, ) -> None:
        super(MaxPoolerAggregator, self).__init__()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_max_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """
        # if mask is not None:
        #     # Simple Pooling layers
        #     input_tensors = replace_masked_values(
        #         input_tensors, mask.unsqueeze(2), -1e7
        #     )

        input_max_pooled = torch.max(input_tensors, 1)[0]

        return input_max_pooled


# 4 DYNAMICROUTINGAGGREGATOR POOLING

class DynamicRoutingAggregator(nn.Module):
    """
    A ``DynamicRoutingAggregator`` is a dynamic routing layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``,
    where not necessarily input_dim == output_dim.

    Parameters
    ----------
    input_dim : ``int``
        the hidden dim of input
    out_caps_num: `` int``
        num of caps
    out_caps_dim: `` int ``
        dim for each cap
    iter_num": `` int ``
        num of iterations
    """

    def __init__(self, input_dim: int,
                 out_caps_num: int,
                 out_caps_dim: int,
                 iter_num: int = 3,
                 output_format: str = "flatten",
                 activation_function: str = "tanh",
                 device=False,
                 shared_fc=None) -> None:
        super(DynamicRoutingAggregator, self).__init__()
        self.input_dim = input_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.iter_num = iter_num
        self.output_format = output_format
        self.activation_function = activation_function
        self.device = device

        if shared_fc:
            self.shared_fc = shared_fc
        else:
            self.shared_fc = nn.Linear(input_dim, out_caps_dim * out_caps_num)

    def forward(self, input_tensors: torch.Tensor,
                mask: torch.Tensor):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        output_format : how to return the output tensor,

        Returns
        -------
        output_tensors : torch.FloatTensor
            if "flatten":
                return tensor of shape ``(batch_size, out_caps_num * out_caps_dim)`` .
            else:
                return tensor of shape ``(batch_size, out_caps_num, out_caps_dim)``
        """

        # shared caps
        batch_size = input_tensors.size()[0]
        num_tokens = input_tensors.size()[1]

        shared_info = self.shared_fc(input_tensors)  # [batch_size, num_tokens, out_caps_dim * out_caps_num]
        # shared_info = torch.tanh(shared_info)   # activation function: tanh, relu?
        if self.activation_function == "tanh":
            shared_info = torch.tanh(shared_info)
        elif self.activation_function == "relu":
            shared_info = F.relu(shared_info)

        shared_info = shared_info.view([-1, num_tokens,
                                        self.out_caps_num,
                                        self.out_caps_dim])

        # prepare mask
        # print("mask: ", mask.size())

        assert len(mask.size()) == 2
        # [bsz, seq_len, 1]
        mask_float = torch.unsqueeze(mask, dim=-1).to(torch.float32)

        B = torch.zeros(
            [batch_size, num_tokens, self.out_caps_num],
            dtype=torch.float32
        ).to(self.device)

        for i in range(self.iter_num):
            mask_tiled = mask.unsqueeze(-1).repeat(1, 1, self.out_caps_num)
            B = B.masked_fill((1 - mask_tiled).byte(), -1e32)

            C = F.softmax(B, dim=2)
            C = C * mask_float  # (batch_size, num_tokens, out_caps_num)
            C = torch.unsqueeze(C, dim=-1)  # (batch_size, num_tokens, out_caps_num, 1)

            weighted_uhat = C * shared_info  # [batch_size, num_tokens, out_caps_num, out_caps_dim]

            S = torch.sum(weighted_uhat, dim=1)  # [batch_size, out_caps_num, out_caps_dim]

            V = squash(S, dim=2)  # [batch_size, out_caps_num, out_caps_dim]
            V = torch.unsqueeze(V, dim=1)  # [batch_size, 1, out_caps_num, out_caps_dim]

            B += torch.sum((shared_info * V).detach(), dim=-1)  # [batch_size, num_tokens, out_caps_num]

        V_ret = torch.squeeze(V, dim=1)  # (batch_size, out_caps_num, out_caps_dim)

        if self.output_format == "flatten":
            V_ret = V_ret.view([
                -1, self.out_caps_num * self.out_caps_dim
            ])
        return V_ret


def squash(input_tensors, dim=2):
    """
    Squashing function
    Parameters
    ----------
    input_tensors : a tensor
    dim: dimensions along which to apply squashing

    Returns
    -------
    squashed : torch.FloatTensor
        A tensor of shape ``(batch_size, num_tokens, input_dim)`` .
    """
    norm = torch.norm(input_tensors, 2, dim=dim, keepdim=True)  # [batch_size, out_caps_num, 1]
    norm_sq = norm ** 2  # [batch_size, out_caps_num, 1]
    s = norm_sq / (1.0 + norm_sq) * input_tensors / torch.sqrt(norm_sq + 1e-8)

    return s
