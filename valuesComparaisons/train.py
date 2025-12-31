import torch
import numpy
import attrs
import time

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from holo.profilers import ProgressBar, Profiler
from holo.prettyFormats import prettyPrint

from datas import HandleClassifDatas, CustomLoader
   

@attrs.frozen
class ResultClassif():
    loss: float
    confusionMatrix: "ConfusionMatrix"
    
    def accuracy(self)->float:
        return self.confusionMatrix.accuracy()
    
    def __str__(self) -> str:
        return f"(loss: {self.loss:.4g}, accuracy: {self.accuracy():.2%})"

@attrs.frozen
class EpochResultClassif():
    epochID: int
    train: ResultClassif
    test: ResultClassif
    
    def __str__(self) -> str:
        return f"Epoch {self.epochID}, train: {self.train}, test: {self.test}"

class HistoryClassification(list[EpochResultClassif]):
    
    def _fig(self, nrows:int):
        axs: list[Axes]
        fig, axs = plt.subplots(ncols=1, nrows=nrows, sharex=True)
        if nrows == 1:
            axs = [axs] # type: ignore
        epoches = list(range(1, 1+len(self)))
        return fig, axs, epoches
    
    def plot(self):
        fig, axs, epoches = self._fig(nrows=2)
        
        axs[0].plot(epoches, [r.train.loss for r in self], label="loss_train")
        axs[0].plot(epoches, [r.test.loss for r in self], label="loss_test")
        axs[0].legend()
        axs[0].set_yscale("log")
        axs[0].grid(True)
        
        axs[1].plot(epoches, [r.train.accuracy() for r in self], label="accuracy_train")
        axs[1].plot(epoches, [r.test.accuracy() for r in self], label="accuracy_test")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.show()
    



class ConfusionMatrix():
    def __init__(self, nbClasses:int) -> None:
        self.matrix = numpy.zeros((nbClasses, nbClasses), dtype=numpy.int32)
        """(rows:predicted, cols:truth)"""
    
    def nb_total(self)->int:
        return int(self.matrix.sum())
    def nb_expected(self, classIndex:int)->int:
        return int(self.matrix[:, classIndex].sum())
    def nb_predicted(self, classIndex:int)->int:
        return int(self.matrix[classIndex, :].sum())
    def nb_truePositive(self, classIndex:int|None)->int:
        if classIndex is not None:
            return int(self.matrix[classIndex, classIndex])
        return int(self.matrix.diagonal().sum())
            
    
    def accuracy(self)->float:
        return self.nb_truePositive(None) / self.nb_total()
    def classExpectedBalance(self, classIndex:int)->float:
        """how much the class was expected"""
        return self.nb_expected(classIndex) / self.nb_total()
    def classPredictedBalance(self, classIndex:int)->float:
        """how much the class was predicted"""
        return self.nb_predicted(classIndex) / self.matrix.sum()
    def classPrecision(self, classIndex:int)->float:
        """when predicted, how much were a good"""
        return  (self.nb_truePositive(classIndex) / self.nb_predicted(classIndex))
    def classHitRate(self, classIndex:int)->float:
        """when expected, how much were a good"""
        return  (self.nb_truePositive(classIndex) / self.nb_expected(classIndex))
    
    def worstK_confusions(self, k:int)->list[tuple[float, int, int]]:
        """return the list of the k worst confusions:
        list of tuple(nbPredictions/totalErrors, truth, predicted)"""
        totalErrs = self.nb_total() - self.nb_truePositive(None)
        return sorted(
            [(float(self.matrix[clPred, clTrue]/totalErrs), clTrue, clPred)
             for clPred in range(self.matrix.shape[0]) for clTrue in range(self.matrix.shape[0])
             if clPred != clTrue],
            reverse=True)[: k]
    
    def step(self, predLabels:list[int], truthLabels:list[int])->None:
        for pred, truth in zip(predLabels, truthLabels):
            self.matrix[pred, truth] += 1


class EpochesTarget():
    def __init__(self, minEpoches:int|None=None, maxEpoches:int|None=None, 
                 minTime:float|None=None, maxTime:float|None=None) -> None:
        self.minEpoches:int|None = minEpoches
        self.maxEpoches:int|None = maxEpoches
        self.minTime:float|None = minTime
        self.maxTime:float|None = maxTime
        assert not all(v is None for v in [minEpoches, maxEpoches, minTime, maxTime])
        
    def start(self)->None:
        self._startTime: float = time.perf_counter()
        self.nbEpochesDone: int = 0
    
    def canDoNextEpoche(self)->bool:
        if (self.minEpoches is not None):
            minEpochesCanStop = (self.nbEpochesDone >= self.minEpoches)
        else: minEpochesCanStop = None
        if (self.maxEpoches is not None):
            maxEpochesNeedStop = (self.nbEpochesDone >= self.maxEpoches)
        else: maxEpochesNeedStop = None
        timeSpent = (time.perf_counter() - self._startTime)
        if (self.minTime is not None):
            minTimeCanStop = (timeSpent >= self.minTime)
        else: minTimeCanStop = None
        if (self.maxTime is not None):
            maxTimeNeedStop = (timeSpent >= self.maxTime)
        else: maxTimeNeedStop = None
        minReached = (minEpochesCanStop is True) or (minTimeCanStop is True)
        minReached = minReached or ((minEpochesCanStop is None) and (minTimeCanStop is None))
        maxReached = (maxEpochesNeedStop is True) or (maxTimeNeedStop is True)
        maxReached = maxReached or ((maxEpochesNeedStop is None) and (maxTimeNeedStop is None))
        #print(f"{minEpochesCanStop=}, {minTimeCanStop=}, {maxEpochesNeedStop=}, {maxTimeNeedStop=}")
        #print(f"{timeSpent=}, {self.nbEpochesDone=}, {minReached=}, {maxReached=}")
        if minReached is False:
            return True # we whant at least the min
        elif maxReached is True:
            return False # => minReached and maxReached (finished)
        else: # => minReached and not maxReached
            return True # need to continue

class TrainerClassif():
    
    def __init__(
            self, model:torch.nn.Module, optimizer:torch.optim.Optimizer,
            criterion:torch.nn.Module, device:torch.device) -> None:
        self.model: torch.nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: torch.nn.Module = criterion
        self.device: torch.device = device
        self.updateEvery: float = (1/20)
        self.history = HistoryClassification()
    
    def train_model_classif(
            self, *, datasHandler:HandleClassifDatas, epoches:int|EpochesTarget)->ResultClassif:
        if isinstance(epoches, int):
            epoches = EpochesTarget(minEpoches=epoches)
        return self.train_model_classif_base(
            datasTrain=datasHandler.train_cLoader(), 
            datasTest=datasHandler.test_cLoader(),
            nbClasses=datasHandler.nbClasses, 
            epoches=epoches)

    def train_model_classif_base(
            self, *, datasTrain: CustomLoader, datasTest:CustomLoader, 
            nbClasses:int, epoches:EpochesTarget)->ResultClassif:
        _prof = Profiler([
            "all", "getBatch", "predict+loss", "backward", "step", 
            "metrics_base", "evaluate", "progressBar", ])
        startEpochID: int = (0 if len(self.history) == 0 else self.history[-1].epochID)
        epoches.start()
        with _prof.mesure("all"):
            epochID: int = startEpochID+1
            while epoches.canDoNextEpoche():
                running_loss = 0.0
                trainConfMatrix = ConfusionMatrix(nbClasses=nbClasses)
                pbar = self._getPBar(len(datasTrain), "train batches")
                self.model.train()
                trainDatasIterator = iter(datasTrain(self.device))
                while True:
                    try:
                        with _prof.mesure("getBatch"):
                            inputs, labels, _ = next(trainDatasIterator)
                    except StopIteration: break
                    self.optimizer.zero_grad()
                    with _prof.mesure("predict+loss"):
                        outputs, loss = self.forwardAndLoss(inputs, labels)
                    with _prof.mesure("backward"):
                        loss.backward()
                    with _prof.mesure("step"):
                        self.optimizer.step()
                    with _prof.mesure("metrics_base"):
                        running_loss += loss.item()
                        predLabels = torch.argmax(outputs.detach(), dim=-1)
                        trainConfMatrix.step(predLabels=predLabels.tolist(), truthLabels=labels.tolist())
                    with _prof.mesure("progressBar"):
                        pbar.step(1)
                self.model.eval()
                with _prof.mesure("evaluate"):
                    testResult = self.eval_model_classif(datas=datasTest, nbClasses=nbClasses, verbose=True)
                meanLoss = (running_loss / len(datasTrain))
                trainResult = ResultClassif(
                    loss=meanLoss, confusionMatrix=trainConfMatrix)
                epoches.nbEpochesDone += 1
                epochID += 1
                self.history.append(EpochResultClassif(epochID, trainResult, testResult))
                print(self.history[-1])
        # show the time it took
        tt = _prof.totalMesure("all")
        times = _prof.totalTimes()
        times["other"] = tt - (sum(times.values()) - tt) # type: ignore
        prettyPrint(times, specificFormats={float: lambda x: f"{x/tt:.2%}"})
        return self.history[-1].train
    
    def eval_model_classif(
            self, *, datas:CustomLoader, nbClasses:int, 
            verbose:bool) -> ResultClassif:
        running_loss = 0.0
        confMatrix = ConfusionMatrix(nbClasses=nbClasses)
        pbar = self._getPBar(len(datas), "test batches")
        self.model.eval()
        for inputs, labels, _ in datas(self.device):
            with torch.no_grad():
                outputs, loss = self.forwardAndLoss(inputs, labels)
            running_loss += loss.item()
            predLabels = torch.argmax(outputs.detach(), dim=-1)
            confMatrix.step(predLabels=predLabels.tolist(), truthLabels=labels.tolist())
            if verbose is True:
                pbar.step()
        meanLoss = (running_loss / len(datas))
        return ResultClassif(loss=meanLoss, confusionMatrix=confMatrix)
    
    def forwardAndLoss(
            self, inputs:torch.Tensor, labels:torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: torch.Tensor = self.model(inputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        return (outputs, loss)

    def _getPBar(self, nb:int, name:str)->ProgressBar:
        return ProgressBar.simpleConfig(
            nb, name, newLineWhenFinished=False, updateEvery=self.updateEvery)
