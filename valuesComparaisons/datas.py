import pathlib
from typing import (
    Callable, Protocol, Iterator, TypedDict, Literal, )

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split



_DatasIterator = Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
"""yield (x, y, idx) tensors that are alredy on the device"""



class DatasIterFuncs(Protocol):
    def __call__(self, loader: DataLoader, device:torch.device) -> _DatasIterator:
        raise NotImplementedError

class CustomLoader():
    __slots__ = ("dataloader", "iterFunc", )
    def __init__(self, dataloader:DataLoader, iterFunc:DatasIterFuncs) -> None:
        self.dataloader = dataloader
        self.iterFunc = iterFunc
    
    def __len__(self)->int:
        return len(self.dataloader)
    
    def __call__(self, device:torch.device)->_DatasIterator:
        return self.iterFunc(self.dataloader, device=device)




class SizedDataset(Dataset):

    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    def iterDataloader(loader: DataLoader, device: torch.device)->_DatasIterator:
        """simplify the way to get the batched datas from dataloader\n
        yield (x, y, idx) tensors that are alredy on the device"""
        raise NotImplementedError


class TargetedClass(int):
    ...


class ClassifDatasetOutput(TypedDict):
    inputs: torch.Tensor
    cls: TargetedClass
    index: int

class ClassifDataloaderOutput(TypedDict):
    inputs: torch.Tensor
    cls: torch.LongTensor
    index: torch.LongTensor

class ValuesClassifDataset(SizedDataset):

    def __init__(self, allValues: torch.Tensor):
        """allValues: (samples, nbValues)"""
        assert allValues.dim() == 2
        self.inputs: torch.Tensor = allValues.cpu().detach()
        self.targets: list[TargetedClass] = \
            list(map(TargetedClass, self.inputs.argmax(dim=1).tolist()))

    @property
    def nbClasses(self)->int:
        return self.inputs.shape[1]

    def __len__(self)->int:
        return self.inputs.shape[0]

    def __getitem__(self, idx)->ClassifDatasetOutput:
        assert isinstance(idx, int)
        idx = int(idx)
        return {'inputs': self.inputs[idx], 'cls': self.targets[idx], "index": idx}

    @staticmethod
    def iterDataloader(loader: DataLoader, device: torch.device)-> _DatasIterator:
        sample: ClassifDataloaderOutput
        for sample in loader:
            yield (sample["inputs"].to(device), sample["cls"].to(device), sample["index"])



class HandleClassifDatas():
    def __init__(self, fullDataset: SizedDataset,
                 name: str, trainProp: float, nbClasses:int,
                 batchSizeTrain: int, batchSizeTest: int) -> None:
        """setup the test/train split and their dataloader based on 
            a given set of images to use (consider that the index are alredy offsetted)"""
        self.name: str = name
        self.nbClasses: int = nbClasses
        self.full_dataset = fullDataset
        nbSamplesTrain = int(len(self.full_dataset) * trainProp)
        self.dataset_train, self.dataset_test = random_split(
            self.full_dataset, lengths=[nbSamplesTrain, (len(self.full_dataset) - nbSamplesTrain)])
        self.datasLoader_train = DataLoader(self.dataset_train, batch_size=batchSizeTrain, shuffle=True, num_workers=0)
        self.datasLoader_test = DataLoader(self.dataset_test, batch_size=batchSizeTest, shuffle=True, num_workers=0)
        print(f"loaded {self.name}(total: {len(self.full_dataset)}), "
              f"train: {len(self.dataset_train)} [{len(self.datasLoader_train)} batches] | "
              f"test: {len(self.dataset_test)} [{len(self.datasLoader_test)} batches]")

    def iterDataloader(self, kind: Literal["train", "test"], device: torch.device):
        loader = (self.datasLoader_train if kind == "train" else self.datasLoader_test)
        return enumerate(self.full_dataset.iterDataloader(loader=loader, device=device))

    def train_cLoader(self)->CustomLoader:
        return CustomLoader(self.datasLoader_train, self.full_dataset.iterDataloader)
    def test_cLoader(self)->CustomLoader:
        return CustomLoader(self.datasLoader_test, self.full_dataset.iterDataloader)

    def setTrainBatchSize(self, batchSize:int)->None:
        self.datasLoader_train = DataLoader(self.dataset_train, batch_size=batchSize, shuffle=True, num_workers=0)
    def setTestBatchSize(self, batchSize:int)->None:
        self.datasLoader_test = DataLoader(self.dataset_test, batch_size=batchSize, shuffle=True, num_workers=0)
        


class BasicNormalValues(HandleClassifDatas):
    fullDataset: ValuesClassifDataset
    
    def __init__(
            self, nbSamples:int, nbClasses:int, trainProp: float,
            batchSizeTrain: int, batchSizeTest: int) -> None:
        fullDataset = ValuesClassifDataset(torch.randn((nbSamples, nbClasses)))
        name = f"BasicNormal[s:{nbSamples},d:{nbClasses}]"
        super().__init__(
            fullDataset=fullDataset, name=name, 
            trainProp=trainProp, nbClasses=fullDataset.nbClasses, 
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)