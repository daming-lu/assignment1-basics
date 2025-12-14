import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def load_data(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # import pdb;pdb.set_trace()
    input_tensor = torch.from_numpy(dataset).to(device)
    num_possible_start = input_tensor.shape[0] - context_length
    
    assert num_possible_start > 0

    start_indices = torch.randint(0, num_possible_start, (batch_size,))

    pre_list = []
    post_list = []
    for start in start_indices.tolist():
        pre = input_tensor[start : start + context_length]
        post = input_tensor[start + 1 : start + 1 + context_length]
        pre_list.append(pre)
        post_list.append(post)

    pre_tensor = torch.stack(pre_list)
    post_tensor = torch.stack(post_list)

    return pre_tensor, post_tensor    
    
