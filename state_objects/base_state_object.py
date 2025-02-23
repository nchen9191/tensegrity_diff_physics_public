import torch.nn
from flatten_dict import flatten, unflatten


class BaseStateObject(torch.nn.Module):

    def __init__(self, name: str):
        super().__init__()
        self.name = name
