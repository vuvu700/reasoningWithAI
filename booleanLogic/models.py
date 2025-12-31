import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import time
import attrs
import numpy

from holo.__typing import Generator, Callable, Literal
from holo.prettyFormats import SingleLinePrinter

from logic import (
    _InstanceArr, _1D_bools, _2D_bools,
    ProblemeLogic_simple, )

_Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

@attrs.frozen
class Results():
    loss: float
    accuracy: float
    
    def __str__(self) -> str:
        return f"(loss: {self.loss:.4g}, accuracy: {self.accuracy:.2%})"

def train_model(
        model:nn.Module, optimizer:torch.optim.Optimizer,
        criterion:"_Criterion", nbEpoches:int, device:torch.device,
        train_dataloader:DataLoader, test_dataloader:DataLoader,
        timeLimit:float|None=None, valLossLimit:float|None=None,
        verbose:bool=True)->list[tuple[Results, Results]]:
    history: list[tuple[Results, Results]] = []
    ptime = time.perf_counter
    tStart = ptime()
    assert nbEpoches > 0
    for epoch in range(nbEpoches):
        running_loss = 0.0
        running_accuracy = 0.0
        nbDone: int = 0
        line = SingleLinePrinter(None)
        tLastLog: float = tStart
        for stepIndex, (inputs, awnsers, idx) in enumerate(
                iterInputAndLabels(train_dataloader, device)):
            if (verbose is True) and (nbDone != 0) \
                    and ((ptime() - tLastLog) > 0.05):
                tLastLog = ptime()
                line.print(f"finished: {stepIndex:_d}/{len(train_dataloader):_d} "
                           f"({stepIndex/len(train_dataloader):.4%}) "
                           f"runningLoss: {running_loss/stepIndex:.4g}, "
                           f"runningAcc: {running_accuracy/nbDone:.2%}")
            model.train(True)
            optimizer.zero_grad()
            outputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(outputs, awnsers)
            loss.backward()
            model.eval()
            optimizer.step()
            running_loss += loss.item()
            running_accuracy += torch.sum((outputs > 0.5) == awnsers).item()
            nbDone += inputs.size(dim=0)
        testResults = eval_model(
            model, device, criterion,
            test_dataloader, verbose=False)
        meanLoss = (running_loss / len(train_dataloader))
        meanAccuracy = (running_accuracy / nbDone)
        history.append((Results(loss=meanLoss, accuracy=meanAccuracy), testResults))
        if verbose is True:
            line.clearLine()
            print(f'Epoch {epoch+1}, train: {history[-1][0]}, test: {testResults}')
        if (timeLimit is not None) and (ptime() - tStart) > timeLimit:
            break
        if (valLossLimit is not None) and (testResults.loss < valLossLimit):
            break
    return history

def eval_model(
        model:nn.Module, device:torch.device, criterion:"_Criterion",
        dataloader:DataLoader, verbose:bool)->Results:
    running_loss = 0.0
    running_accuracy = 0.0
    nbDone: int = 0
    line = SingleLinePrinter(None)
    model.eval()
    for stepIndex, (inputs, awnsers, idx) in enumerate(
            iterInputAndLabels(dataloader, device=device)):
        if verbose is True:
            line.print(f"step: {stepIndex:_d}/{len(dataloader):_d} ({stepIndex/len(dataloader):.4%})")
        with torch.no_grad():
            outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, awnsers)
        running_loss += loss.item()
        running_accuracy += torch.sum((outputs > 0.5) == awnsers).item()
        nbDone += inputs.size(dim=0)
    if verbose is True:
        line.clearLine()
    meanLoss = (running_loss / len(dataloader))
    meanAccuracy = (running_accuracy / nbDone)
    return Results(loss=meanLoss, accuracy=meanAccuracy)



class SimpleLogicDataset(Dataset):
    def __init__(self, inputs:_InstanceArr, awnsers:_1D_bools):
        self.inputs = inputs
        self.awnsers = awnsers
    
    def __len__(self)->int:
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx = int(idx)
        return (self.inputs[idx], self.awnsers[idx], idx)

    @staticmethod
    def genRandomFromProblem(
            pb:ProblemeLogic_simple, nbSamples:int)->"SimpleLogicDataset":
        return SimpleLogicDataset(*pb.generateBatch(nbSamples))

    @staticmethod
    def genAllFromProblem(
            pb:ProblemeLogic_simple)->"SimpleLogicDataset":
        return SimpleLogicDataset(*pb.generateCombiations())
    
    def split(self, trainProp:float):
        trainSize = int(trainProp * len(self))
        valSize = len(self) - trainSize
        subsets = random_split(self, [trainSize, valSize])
        assert len(subsets) == 2
        return (subsets[0], subsets[1])
    
    def getLoaders(
            self, trainProp:float,
            trainBatchSize:int, valBatchSize:int,
            )->tuple[DataLoader, DataLoader]:
        trainSplit, valSplit = self.split(trainProp=trainProp)
        trainLoader = DataLoader(trainSplit, shuffle=True, batch_size=trainBatchSize)
        valLoader = DataLoader(valSplit, shuffle=True, batch_size=valBatchSize)
        return (trainLoader, valLoader)
        
        

def iterInputAndLabels(
        loader:DataLoader, device:torch.device,
        )->Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    for x, y, idx in loader:
        assert isinstance(x, torch.Tensor) and (x.ndim == 2) and (x.dtype == torch.bool)
        assert isinstance(y, torch.Tensor) and (y.ndim == 1) and (y.dtype == torch.bool)
        assert isinstance(idx, torch.Tensor) and (idx.ndim == 1) and (idx.dtype == torch.int64)
        assert x.size(0) == y.size(0) == idx.size(0)
        x = x.to(torch.float32).to(device)
        y = y[:, None].to(torch.float32).to(device)
        yield (x, y, idx)

_Activ = Literal["relu", "sigmoid", "tanh", "none"]


def getActiv(name: _Activ):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"unknown activation: {name!r}")


class DenseModelsGeneric(nn.Module):
    def __init__(self, layers: list[tuple[int, int, _Activ]], 
                 optim:type[torch.optim.Optimizer], lr:float,
                 criterion:"_Criterion"):
        super().__init__()
        self._layers = nn.Sequential()
        for (d1, d2, activ) in layers:
            self._layers.extend([
                nn.Linear(d1, d2), getActiv(activ)])

        self.optim = optim(self._layers.parameters(), lr=0.001) # type: ignore
        self.loss: _Criterion = criterion

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self._layers(x)
    
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def countParameters(self)->int:
        return sum(int(numpy.prod(params.size())) 
                   for params in self._layers.parameters())

class DenseSkipedModelsGeneric(nn.Module):
    def __init__(self, layers: list[tuple[int, int, _Activ]], 
                 optim:type[torch.optim.Optimizer], lr:float,
                 criterion:"_Criterion"):
        super().__init__()
        self._layers: list[tuple[nn.Module, nn.Module]] = []
        for i, (d1, d2, activ) in enumerate(layers):
            lin = nn.Linear(d1, d2)
            act = getActiv(activ)
            self.add_module(f"Linear({i})", lin)
            self.add_module(f"{act._get_name()}({i})", act)
            self._layers.append((lin, act))

        self.optim = optim(self.getModelParams(), lr=0.001) # type: ignore
        self.loss: _Criterion = criterion

    def forward(self, x: torch.Tensor)->torch.Tensor:
        for layers in self._layers:
            x_out = layers[1](layers[0](x))
            if x_out.shape == x.shape:
                x = x + x_out
            else: x = x_out
        return x
    
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        return super().__call__(x)
    
    def getModelParams(self):
        for grp in self._layers:
            for layer in grp:
                yield from layer.parameters()
    
    def countParameters(self)->int:
        return sum(int(numpy.prod(param.size()))
                   for param in self.getModelParams())