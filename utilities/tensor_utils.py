from typing import Tuple, Optional

import torch


def zeros(shape: Tuple,
          ref_tensor: Optional[torch.Tensor] = None):
    dtype = ref_tensor.dtype
    device = ref_tensor.device

    return torch.zeros(shape, dtype=dtype, device=device)


def ones(shape: Tuple,
          ref_tensor: Optional[torch.Tensor] = None):

    dtype = ref_tensor.dtype
    device = ref_tensor.device

    return torch.ones(shape, dtype=dtype, device=device)
