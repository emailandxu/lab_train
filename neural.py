import numpy as np
from neural_funcs import *


def add(weights, inputs):
    """# w1p1 + w2p2 + ..."""
    assert len(weights) == len(inputs)
    s = 0
    for index, weight in enumerate(weights):
        result = weight * inputs[index]
        s += result
        # print(index,result)
    return s

def neural_output(weights, outputs):
    """weight sum and through the 
    active function add bias"""
    s = add(weights, outputs)
    a = func_sigmoid(s)
    outputs = func_bias(a, 1)
    return outputs

def loss():
    pass
print(neural_output(
    [3.1215926, 1.234567],
    [0, 1]
))
