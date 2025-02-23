import json
import shutil
from pathlib import Path

import torch

from utilities import torch_quaternion


def get_num_precision(precision: str) -> torch.dtype:
    if precision.lower() == 'float':
        return torch.float
    elif precision.lower() == 'float16':
        return torch.float16
    elif precision.lower() == 'float32':
        return torch.float32
    elif precision.lower() == 'float64':
        return torch.float64
    else:
        print("Precision unknown, defaulting to float16")
        return torch.float
