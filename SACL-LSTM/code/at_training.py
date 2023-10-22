import logging
import re
import torch

logger = logging.getLogger(__name__)



class FGM(object):
    """Reference: https://arxiv.org/pdf/1605.07725.pdf"""

    def __init__(self,
                 model,
                 emb_names=['word_embeddings', "encoder.layer.0"],
                 epsilon=1.0):
        self.model = model
        self.emb_names = emb_names
        self.epsilon = epsilon
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon_var=None):
        """Add adversity."""

        if epsilon_var is None:
            epsilon = self.epsilon
        else:
            epsilon = self.epsilon + epsilon_var * torch.randn(1).cuda()  # v1

        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        """ restore embedding """
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]





class PGD(object):
    """Reference: https://arxiv.org/pdf/1706.06083.pdf"""

    def __init__(self,
                 model,
                 emb_names=['word_embeddings', "encoder.layer.0"],
                 epsilon=1.0,
                 alpha=0.3):
        self.model = model
        self.emb_names = emb_names
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.alpha * param.grad / norm
                    param.data.add_(r_adv)
                    param.data = self.project(name, param.data)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r_adv = param_data - self.emb_backup[param_name]
        if torch.norm(r_adv) > self.epsilon:
            r_adv = self.epsilon * r_adv / torch.norm(r_adv)
        return self.emb_backup[param_name] + r_adv

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]
