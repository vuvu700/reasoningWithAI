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
        assert self.nbInputs < 25
        tmp = numpy.arange(2**self.nbInputs, dtype=numpy.uint32)
        inputs = ((tmp[:, None] & (1 << numpy.arange(self.nbInputs))) > 0)
        return self.getBatch(inputs)

def funcSimple_1(inputs:_InstanceArr)->_1D_bools:
    """truth balance: 62.5%, expect 3 inputs (8 samples)"""
    return (inputs[:, 0] & inputs[:, 1]) | inputs[:, 2]
PROBLEME_1 = ProblemeLogic_simple(funcSimple_1, nbInputs=3)


def funcSimple_2(inputs:_InstanceArr)->_1D_bools:
    """truth balance: 34.4%, expect 5 inputs (32 samples)"""
    a,b,c,d,e = (inputs[:, idx] for idx in range(5))
    return ((a & b) | (c | d)) & ((d | e) & a)
PROBLEME_2 = ProblemeLogic_simple(funcSimple_2, nbInputs=5)

def funcSimple_3(inputs:_InstanceArr)->_1D_bools:
    """truth balance: 43.7%, expect 5 inputs (32 samples)"""
    a,b,c,d,e = (inputs[:, idx] for idx in range(5))
    return ((a ^ ~b) | (c | d)) & ((d & e) ^ ~a)
PROBLEME_3 = ProblemeLogic_simple(funcSimple_3, nbInputs=5)

def funcSimple_4(inputs:_InstanceArr)->_1D_bools:
    """truth balance: 47.7%, expect 10 inputs (1024 samples)"""
    a,b,c,d,e,f,g,h,i,j = (inputs[:, idx] for idx in range(10))
    return ((a | b ^ c) ^ (d & e | f)) ^ (g & h | i) & j
PROBLEME_4 = ProblemeLogic_simple(funcSimple_4, nbInputs=10)

def funcSimple_5(inputs: _InstanceArr) -> _1D_bools:
    """truth balance: 49.9%, expect 20 inputs (1,048,576 samples)"""
    # NOTE: couldn't learn on it
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t = (
        inputs[:, idx] for idx in range(20))
    return (
        (((a | b ^ c) ^ (d & e | f)) ^ ((g & h) | (i ^ j)))
        ^ (((k | l ^ m) ^ (n & o | p)) ^ ((q & r) | (s ^ t))))
PROBLEME_5 = ProblemeLogic_simple(funcSimple_5, nbInputs=20)

def funcSimple_6(inputs:_InstanceArr)->_1D_bools:
    """truth balance: 51.9%, expect 20 inputs (1,048,576 samples)"""
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t = (
        inputs[:, idx] for idx in range(20))
    part1 = ((a | b ^ c) ^ (d & e | f))
    part2 = ((g & h | i) ^ (j | k ^ l))
    part3 = ((m ^ n & o) | (p & q ^ r))
    part4 = ((s | t) ^ (a & j | r))
    return ((part1 ^ part2) & (part3 | part4)) ^ (k & m | p)
PROBLEME_6 = ProblemeLogic_simple(funcSimple_6, nbInputs=20)


def funcSimple_7(inputs: _InstanceArr) -> _1D_bools:
    """truth balance: 49.7%, expect 20 inputs (1,048,576 samples)"""
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t = (
        inputs[:, idx] for idx in range(20))
    return ((((a & b) ^ (c | d ^ e)) ^ ((f | g) ^ (h & i | j)))
            ^ (((k ^ l & m) | (n ^ o | p)) ^ ((q | r ^ s) & (t | a))))
PROBLEME_7 = ProblemeLogic_simple(funcSimple_7, nbInputs=20)

def funcSimple_8(inputs: _InstanceArr) -> _1D_bools:
    """truth balance: 30.5%, expect 20 inputs (1,048,576 samples)"""
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t = (
        inputs[:, idx] for idx in range(20))
    L1 = (a ^ b) & (c | d)
    L2 = (e | f) ^ (g & h)
    L3 = (i ^ j) | (k & L1)
    L4 = (l & m) ^ (n | L2)
    L5 = (o | p) ^ (L3 & L4)
    L6 = ((q ^ r) & (s | t)) ^ (L5 | L1)
    L7 = (L2 ^ L6) & ((a | n) ^ (e & p))
    L8 = (L7 | (b & h)) ^ ((j | q) & (L3 ^ g))
    L9 = (L8 & (L4 | i)) ^ (L5 ^ (t & c))
    L10 = ((L9 | L7) ^ (L2 & k)) | ((r ^ f) & (o | e))
    L11 = ((L10 ^ L8) & (L6 | L3)) ^ ((p & q) | (a ^ m))
    return (L11 ^ (L9 | L4)) & ((L7 & L5) | (L10 ^ L1))
PROBLEME_8 = ProblemeLogic_simple(funcSimple_8, nbInputs=20)

def funcSimple_9(inputs: _InstanceArr) -> _1D_bools:
    """truth balance: 44.7%, expect 20 inputs (1,048,576 samples)"""
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t = (inputs[:, idx] for idx in range(20))
    L1 = (a ^ b) | (c & d)
    L2 = (e & f) ^ (g | h)
    L3 = (i ^ j) & (k | l)
    L4 = (m | n) ^ (o & p)
    L5 = (L1 & L2) ^ (L3 | L4)
    L6 = ((q ^ r) | (s & t)) ^ ((a & g) | (m ^ j))
    L7 = (L5 ^ L6) & ((d | n) ^ (e & k))
    return ((L7 | L2) ^ (L5 & L4)) ^ ((h & q) | (b ^ t))
PROBLEME_9 = ProblemeLogic_simple(funcSimple_9, nbInputs=20)