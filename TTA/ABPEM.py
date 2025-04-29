import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math
import random
import numpy as np


class Cache:
    def __init__(self, size) -> None:
        self.features = []
        self.sample_attns = []
        self.labels = []
        self.size = size
    
    def update(self):
        if len(self.features) > self.size:
            self.features = self.features[1:]
        if len(self.sample_attns) > self.size:
            self.sample_attns = self.sample_attns[1:]
        if len(self.labels) > self.size:
            self.labels = self.labels[1:]
    
    def sample_feature(self):
        idx = random.randint(0, len(self.features) - 1)
        return self.features[idx]

def kl_div(p_mean, p_std, q_mean, q_std):
    return torch.log(q_std / p_std) + (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2) - 0.5

class READ(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

    def forward(self, x, adapt_flag, labels=None):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler, labels=labels)
            else:
                args = self.args
                with torch.no_grad():
                    outputs, attn_raw = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode, args=args)
                loss = (0, 0, 0)
                outputs = (outputs, outputs)

        return outputs, loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler, labels=None):
    """Forward and adapt model on batch of data.
    Compute loss function (Eq. 7) based on the model prediction, take gradients, and update params.
    """
    args.counter += 1
    with autocast():
        # forward (x0: audio, B x L x 128, x1: video, B x 3 x H x W)
        outputs, attn_raw, cached_av_feat = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode, args=args, stage='adapt', labels=labels, ret_av_feat=True, output_embedding=True)
    # adapt
    # p_sum = outputs.softmax(dim=-1).sum(dim=-2)
    # loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()    
    if args.fix_loss_bal:
        p_sum = outputs.softmax(dim=-1).mean(dim=-2)
        loss_bal = - (p_sum * p_sum.log()).sum()
    else:
        p_sum = outputs.softmax(dim=-1).sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

    pred = outputs.softmax(dim=-1)
    ent_all = -pred * torch.log(pred + 1e-6)
    ent_idx = torch.sort(pred, dim=-1, descending=True)[1]
    prime_ent = torch.gather(ent_all, dim=-1, index=ent_idx[:, :args.prime_ent_k]).sum(dim=-1)

    loss_ra = prime_ent.mean()

    attn = attn_raw.softmax(dim=-1)
    attn_raw_a2a = attn_raw[:, :512, :512]
    attn_raw_v2v = attn_raw[:, 512:, 512:]
    attn_raw_a2v = attn_raw[:, :512, 512:]
    attn_raw_v2a = attn_raw[:, 512:, :512]
    attn_raw_a2a_mean = attn_raw_a2a.mean(dim=(1, 2)).detach()
    attn_raw_v2v_mean = attn_raw_v2v.mean(dim=(1, 2)).detach()
    attn_raw_a2v_mean = attn_raw_a2v.mean(dim=(1, 2))
    attn_raw_v2a_mean = attn_raw_v2a.mean(dim=(1, 2))
    attn_raw_a2a_std = attn_raw_a2a.std(dim=(1, 2)).detach()
    attn_raw_v2v_std = attn_raw_v2v.std(dim=(1, 2)).detach()
    attn_raw_a2v_std = attn_raw_a2v.std(dim=(1, 2))
    attn_raw_v2a_std = attn_raw_v2a.std(dim=(1, 2))
    kld = kl_div(attn_raw_v2a_mean, attn_raw_v2a_std, attn_raw_a2a_mean, attn_raw_a2a_std).mean() + kl_div(attn_raw_a2v_mean, attn_raw_a2v_std, attn_raw_v2v_mean, attn_raw_v2v_std).mean()

    loss_attn = kld
    

    loss = loss_ra * args.ent_weight - loss_bal * args.bal_weight + loss_attn * args.attn_weight
    
    optimizer.zero_grad()
    if not (args.ent_weight == 0 and args.bal_weight == 0 and args.attn_weight == 0):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


    with torch.no_grad():
        with autocast():
        # forward
            outputs2, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode, args=args, stage='test', labels=labels, av_feat=cached_av_feat)

    return (outputs, outputs2), (loss_ra.item(), loss_bal.item(), loss_attn.item())

requires_grad_list = [
    'module.blocks_u.0.attn.qa', 'module.blocks_u.0.attn.ka', 'module.blocks_u.0.attn.va', 
    # 'module.blocks_u.0.attn.qv', 'module.blocks_u.0.attn.kv', 'module.blocks_u.0.attn.vv'
]

def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_fusion_qkv = []
    names_fusion_qkv = []

    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")

    return params_fusion_qkv, names_fusion_qkv


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if nm in requires_grad_list:
            m.requires_grad_(True)

    return model