""" Optimizers class """
import re

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW



def get_group_parameters(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
def func(s):
    s = re.sub(r'\d+.','',s)
    return s
def get_group_parameters2(model,args):
    no_decay = ["bias", "LayerNorm.weight"]
    other=["user_fuse_layer","agent_fuse_layer"]
    params = list(model.named_parameters())

    no_main=no_decay+other
    param_group = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_main)], 'weight_decay': 1e-2, 'lr': args.lr},
        {'params': [p for n, p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0, 'lr': args.lr},
        {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0, 'lr': 1e-3},
        {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2, 'lr': 1e-3},
    ]

    return param_group




def get_group_parameters_other(model,args):
    no_decay = ["bias", "LayerNorm.weight"]
    other=["role_aware"]
    params = list(model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0, 'lr': 1e-3},
        {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2, 'lr': 1e-3},
    ]

    return param_group

def build_optim_bart(args, model):
    parm_group = get_group_parameters(model)
    optimer = AdamW(parm_group, lr=args.lr)
    return optimer

def build_optim_bart2(args, model):
    parm_group = get_group_parameters2(model,args)
    optimer = AdamW(parm_group,lr=args.lr)
    return optimer


def build_optim_bart_all(args, model):
    main_parm_group = get_group_parameters_main(model,args)
    other_param_group=get_group_parameters_other(model,args)
    main_optimer = AdamW(main_parm_group)
    other_optimer = AdamW(other_param_group)
    return main_optimer,other_optimer
