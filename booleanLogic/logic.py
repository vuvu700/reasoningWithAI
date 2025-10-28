from holo.__typing import Callable
import numpy


_1D_bools = numpy.ndarray[tuple[int], numpy.dtype[numpy.bool_]]
_2D_bools = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool_]]

_InstanceArr = _2D_bools
_LogicFuncSimple = Callable[[_InstanceArr], _1D_bools]
_LogicFuncComplex = Callable[[_InstanceArr], _2D_bools]
_XyModel = tuple[_InstanceArr, _1D_bools]

class ProblemeLogic_simple():
    def __init__(self, logicFunc:"_LogicFuncSimple", nbInputs:int) -> None:
        self.func: "_LogicFuncSimple" = logicFunc
        self.nbInputs: int = nbInputs
    
    def getBatch(self, inputs:_InstanceArr)->_XyModel:
        return (inputs, self.func(inputs))
    
    def generateBatch(self, batchSize:int)->_XyModel:
        inputs: _InstanceArr = (
            numpy.random.random((batchSize, self.nbInputs)) < 0.5)
        return self.getBatch(inputs)

    def generateCombiations(self)->_XyModel:
        assert self.nbInputs < 16
        tmp = numpy.arange(2**self.nbInputs, dtype=numpy.uint32)
        inputs = ((tmp[:, None] & (1 << numpy.arange(self.nbInputs))) > 0)
        return self.getBatch(inputs)

def funcSimple_1(inputs:_InstanceArr)->_1D_bools:
    """formula: ((a & b) | c) -> expect 3 inputs"""
    return (inputs[:, 0] & inputs[:, 1]) | inputs[:, 2]
PROBLEME_1 = ProblemeLogic_simple(funcSimple_1, nbInputs=3)


def funcSimple_2(inputs:_InstanceArr)->_1D_bools:
    """formula: ((a & b) | (c | d)) & ((d | e) & a) -> expect 5 inputs"""
    a = inputs[:, 0]; b = inputs[:, 1]; c = inputs[:, 2]
    d = inputs[:, 3]; e = inputs[:, 4]
    return ((a & b) | (c | d)) & ((d | e) & a)
PROBLEME_2 = ProblemeLogic_simple(funcSimple_2, nbInputs=5)

def funcSimple_3(inputs:_InstanceArr)->_1D_bools:
    """formula: ((a ^ ~b) | (c | d)) & ((d & e) ^ ~a) -> expect 5 inputs"""
    a = inputs[:, 0]; b = inputs[:, 1]; c = inputs[:, 2]
    d = inputs[:, 3]; e = inputs[:, 4]
    return ((a ^ ~b) | (c | d)) & ((d & e) ^ ~a)
PROBLEME_3 = ProblemeLogic_simple(funcSimple_3, nbInputs=5)