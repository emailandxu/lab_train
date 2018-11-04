import math
from pdb import set_trace

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
    return (math.exp(input_)-math.exp(-input_))/(math.exp(input_)+math.exp(-input_))

def func_bias(input_,b):
    return input_ + b*1