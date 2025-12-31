import sys
sys.path.append("..")

from generics.modelsParamaters import ModelTreeNode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import time
import attrs
import numpy

from holo.__typing import Generator, Callable, Literal
from holo.prettyFormats import SingleLinePrinter

_Activ = Literal["relu", "sigmoid", "tanh", "none"]


def getActiv(name: _Activ):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "none":
        return None
    else:
        raise ValueError(f"unknown activation: {name!r}")


class DenseModelsGeneric(nn.Module):
    def __init__(self, layers: list[tuple[int, int, _Activ]], 
                 optim:type[torch.optim.Optimizer], lr:float,
                 criterion:torch.nn.Module):
        super().__init__()
        self._layers = nn.Sequential()
        for (d1, d2, activ) in layers:
            self._layers.append(nn.Linear(d1, d2))
            activ = getActiv(activ)
            if activ is not None:
                self._layers.append(activ)

        self.optim = optim(self._layers.parameters(), lr=0.001) # type: ignore
        self.loss: torch.nn.Module = criterion

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self._layers(x)
    
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def countParameters(self)->int:
        return sum(int(numpy.prod(params.size())) 
                   for params in self._layers.parameters())
