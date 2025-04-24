import re
import numpy as np
import random
import time
from torch.nn.init import *

# Count the number of trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create a character-level tokenizer that handles special link tokens
def char_tokenizer():
    def tokenizer(sent):
        sent=sent.lower()
        spans=re.split(r'(##<link##[1-9]\d*##link>##)', sent)
        tokens=[]
        for span in spans:
            if '##<link##' in span and '##link>##' in span:
                tokens += ['[Link]']
            else:
                tokens += list(''.join(span.split()))
        return tokens
    return tokenizer

# Create a character-level detokenizer
def char_detokenizer():
    def detokenizer(tokens):
        return ''.join(tokens)
    return detokenizer

# Get current timestamp in milliseconds
def get_ms():
    return time.time() * 1000

# Initialize random seeds for reproducibility
def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Initialize model parameters using Xavier uniform initialization
def init_params(model, escape=None):
    for name, param in model.named_parameters():
        if escape is not None and escape in name:
            print('no_init', name, param.size())
            continue
        print('init', name, param.size())
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        # if 'bias' in name:
        #     constant_(param.data, 0)
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()

# Freeze all parameters in the model
def freeze_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
        print('freeze_params', name, param.size())

# Unfreeze all parameters in the model
def unfreeze_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
        print('unfreeze_params', name, param.size())

