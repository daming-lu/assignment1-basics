import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor



def save_checkpoint(model, optimizer, iteration, out):
    # import pdb;pdb.set_trace()
    model_optimizer_dict = {}
    model_optimizer_dict['model_state_dict'] = model.state_dict()
    model_optimizer_dict['optimizer_state_dict'] = optimizer.state_dict()
    model_optimizer_dict['iteration'] = iteration
    torch.save(model_optimizer_dict, out)



def load_checkpoint(src, model, optimizer):
    model_optimizer_dict = torch.load(src)
    model.load_state_dict(model_optimizer_dict['model_state_dict'])
    optimizer.load_state_dict(model_optimizer_dict['optimizer_state_dict'])
    return model_optimizer_dict['iteration']
