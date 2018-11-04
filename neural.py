import numpy as np
import math

# w1p1 + w2p2 + ...
def add(weights,inputs):
    assert len(weights) == len(inputs)
    s = 0 
    for index,weight in enumerate(weights):
        result = weight * inputs[index]
        s += result
        # print(index,result)
    return s

def func_liner(input_):
    return input_

def func_sigmoid(input_):
    return 1/(1+math.exp(-input_))

def func_relu(input_):
    if input_ < 0:
        return 0
    elif input_ < 1:
        return input_
    else:
        return 1

def func_tanh(input_):
    pass

s = add([3,2,3],[2,4,7])
print(s)
print("Sigmoid",func_sigmoid(s))
print("Relu",func_relu(s))

