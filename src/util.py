from torch import Tensor
from typing import Union

def standardize(input: Tensor, mean: Union[Tensor, None] = None, std: Union[Tensor, None] = None) -> Tensor:
    if mean is None:
        mean = input.mean(dim = 0, keepdim = True)

    if std is None:
        std = input.std(dim = 0, keepdim = True)
    return (input - mean) / std

def unstandardize(input: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return input * std + mean


