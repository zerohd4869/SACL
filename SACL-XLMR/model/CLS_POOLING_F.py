#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("..")
import torch
import torch.nn as nn
from utils.dice_loss import DiceLoss
from utils.focal_loss import FocalLoss
from transformers import AutoModel, AutoConfig
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
from pytorch_metric_learning.distances import DotProductSimilarity


class ClassifierModel(nn.Module):
    def __init__(self, args, class_weights, tokenizer):
        super(ClassifierModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.class_weights = class_weights
        self.config = AutoConfig.from_pretrained(self.args.pretrain_model_path, gradient_checkpointing=True)
        self.roberta = AutoModel.from_pretrained(
            args.pretrain_model_path,
            from_tf=bool(".ckpt" in self.args.pretrain_model_path),
            config=self.config,
            cache_dir=self.args.cache_dir)
        self.roberta.resize_token_embeddings(len(tokenizer))
        self.class_nums = len(class_weights)
        self.fc = nn.Linear(self.config.hidden_size, self.class_nums)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)

        if self.args.mutisample_dropout == True:
            self.dropout_ops = nn.ModuleList(
                nn.Dropout(self.args.dropout_rate) for _ in range(self.args.dropout_num)
            )
        else:
            self.dropout_ops = nn.Dropout(self.args.dropout)
        self._init_weights(self.layer_norm)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def loss_fn(self, logits, labels, adv_flag=False):
        # todo weight
        if self.args.use_class_weights:
            weight = self.class_weights
        else:
            weight = None

        if self.args.loss_fct_name == "CrossEntropy":
            loss_fct = nn.CrossEntropyLoss(
                weight=None,
            )
        elif self.args.loss_fct_name == "Focal":
            loss_fct = FocalLoss(
                gamma=self.args.focal_loss_gamma,
                alpha=weight,
                class_num=2
                # reduction="mean"
            )
        elif self.args.loss_fct_name == "Dice":
            loss_fct = DiceLoss(
                with_logits=True,
                smooth=1.0,
                ohem_ratio=0.8,
                alpha=0.01,
                square_denominator=True,
                index_label_position=True,
                reduction="mean"
            )
        else:
            raise ValueError("unsupported loss function: {}".format(self.args.use_contrastive_loss))

        loss = loss_fct(logits, labels.long())
        if self.args.contrastive_loss_flag:

            if self.args.contrastive_loss == "NTXent":
                loss_fct_contrast = NTXentLoss(
                    temperature=self.args.contrastive_temperature,
                    distance=DotProductSimilarity(),
                )
            elif self.args.contrastive_loss == "SupCon":
                loss_fct_contrast = SupConLoss(
                    temperature=self.args.contrastive_temperature,
                    distance=DotProductSimilarity(),
                )
            else:
                raise ValueError("unsupported contrastive loss function: {}".format(self.args.use_contrastive_loss))

            if self.args.what_to_contrast == "sample":
                embeddings = logits
                labels = labels.view(-1)
            elif self.args.what_to_contrast == "sample_and_class_embeddings":
                embeddings = torch.cat(
                    [logits, self.fc.weight],
                    dim=0
                )
                labels = torch.cat(
                    [
                        labels.view(-1),
                        torch.arange(0, self.args.num_labels_level_2).to(self.args.device)
                    ],
                    dim=-1
                )
            else:
                raise ValueError("unsupported contrastive features: {}".format(self.args.what_to_contrast))

            contra_loss = loss_fct_contrast(
                embeddings,
                labels
            )
            print("adv_flag: {}, ce_loss: {}, cl_loss: {}".format(adv_flag,loss,contra_loss))
            if adv_flag:
                loss = loss + self.args.contrastive_loss_weight2 * contra_loss
            else:
                loss = loss + self.args.contrastive_loss_weight * contra_loss
        return loss

    def model_mutisample_dropout(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.fc(out)
            else:
                temp_out = dropout_op(x)
                temp_logits = self.fc(temp_out)
                logits += temp_logits

        if self.args.dropout_action:
            logits = logits / self.args.dropout_num

        return logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, adv_flag=False):

        # BaseModelOutput: (last_hidden_state, hidden_states, attentions)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)

        if "deberta" in self.args.pretrain_model_path:
            outputs = outputs["last_hidden_state"]
            pooler_output = outputs[:, 0]  # [CLS]
        else:
            pooler_output = outputs[1]

        pooler_output = self.layer_norm(pooler_output)
        logits = self.model_mutisample_dropout(pooler_output)

        if labels is not None:
            loss = self.loss_fn(logits, labels, adv_flag)
        else:
            loss = 0

        if "deberta" in self.args.pretrain_model_path:
            output = (logits,) + (outputs[1:],)
        else:
            output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
