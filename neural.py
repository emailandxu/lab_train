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
        raise e
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

def batch_loss(y,weights, bias, inputs, outputs):
    """update weights and bias of a layer use batch gradient descend
    @params:
    inputs: 2-d list
    outputs: 1-d list
    y: 1-d list
    weights: 1-d list
    bias:
    """
    assert type(y) is list and type(outputs) is list
    for index_attr,weight in enumerate(weights):
        errors_for_weights = [(y[index_record]-output_)*inputs[index_record][index_attr] for index_record, output_ in enumerate(outputs)]
        mean_e_for_weights = sum(errors_for_weights)/len(errors_for_weights)
        weights[index_attr] += mean_e_for_weights
    
    errors_for_bias = [y[index_record]-output_ for index_record, output_ in enumerate(outputs)]
    mean_e_for_bias = sum(errors_for_bias)/len(errors_for_bias)
    bias += mean_e_for_bias
    
    
    return weights,bias


def batch_loss_no_x(y,weights, bias, inputs, outputs):
    """update weights and bias of a layer use batch gradient descend
    @params:
    inputs: 2-d list
    outputs: 1-d list
    y: 1-d list
    weights: 1-d list
    bias: float
    """
    assert type(y) is list and type(outputs) is list
    
    errors = [y[index_record]-output_ for index_record, output_ in enumerate(outputs)]
    mean_e = sum(errors)/len(errors)
    bias += mean_e

    return [weight+mean_e for weight in weights],bias



def run_a_loop():
    weights = [1.232434, -3.754862, -1.342545]
    # atrribute:
    # color 1 red 0 yellow
    # shape 1 circle 0 rect
    # smell 1 apple 0 banana
    # y: 0 apple, 1 banana
    multi_inputs = [[1,1,0],[0,0,1]]
    y = [0,1]
    bias = 1
    print(weights,bias)
    # loop times
    for i in range(1000):
        # a loop update weights by all records
        for index,single_inputs in enumerate(multi_inputs):
            outputs = neural_output(weights,bias, single_inputs)
            weights,bias = loss(y[index],weights,bias,single_inputs,outputs)
            print(outputs)
    
    print(weights,bias)

def run_a_loop_use_bath_loss():
    weights = [1.232434, -3.754862, -1.342545]
    # atrribute:
    # color 1 red 0 yellow
    # shape 1 circle 0 rect
    # smell 1 apple 0 banana
    # y: 0 apple, 1 banana
    multi_inputs = [[1,1,0],[0,0,1]]
    y = [0,1]
    bias = 1
    print(weights,bias)
    # loop times
    for i in range(1000):
        # a loop update weights by all records
        outputs = [neural_output(weights,bias,single_inputs) for single_inputs in multi_inputs]
        weights,bias = batch_loss(y,weights,bias,multi_inputs,outputs)
        print(outputs)
    
    print(weights,bias)
    print([add(weights,input_)+bias for input_ in multi_inputs])


def run_a_loop_use_bath_loss_no_x():
    weights = [1.232434, -3.754862, -1.342545]
    # atrribute:
    # color 1 red 0 yellow
    # shape 1 circle 0 rect
    # smell 1 apple 0 banana
    # y: 0 apple, 1 banana
    multi_inputs = [[1,1,0],[0,0,1]]
    y = [0,1]
    bias = 1
    print(weights,bias)
    # loop times
    for i in range(1000):
        # a loop update weights by all records
        outputs = [neural_output(weights,bias,single_inputs) for single_inputs in multi_inputs]
        weights,bias = batch_loss_no_x(y,weights,bias,multi_inputs,outputs)
        print(outputs)
    
    print(weights,bias)

run_a_loop_use_bath_loss()