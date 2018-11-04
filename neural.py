import numpy as np
from neural_funcs import *
from pdb import set_trace

def add(weights, inputs):
    """w1p1 + w2p2 + ..."""
    assert len(weights) == len(inputs)
    s = 0
    for index, weight in enumerate(weights):
        result = weight * inputs[index]
        s += result
        # print(index,result)
    return s


def neural_output(weights, bias,outputs):
    """weight summing plus bias and put the result 
    into the active function, opreations for forward 
    propogation """
    s = add(weights, outputs)
    b = func_bias(s, bias)
    try:
        a = func_sigmoid(b)
    except OverflowError as e:
        print("overflow ",b,bias)
    return a


def loss(y, weights, bias, inputs, outputs):
    """update weights and bias of a layer 
    """
    # error 误差
    e = y - outputs
    for index, input_ in enumerate(inputs):
        # update each value of weights
        weights[index] += e * input_
    # update bias value
    bias += e
    return weights, bias

def run_a_loop():
    y = 0.5
    weights = [1.232434, -3.754862, -1.342545]
    inputs = [1,-1,0]
    bias = 1
    print(weights,bias)
    for i in range(1000):
        outputs = neural_output(weights,bias, inputs)
        weights,bias = loss(y,weights,bias,inputs,outputs)
        print(outputs)
    

run_a_loop()