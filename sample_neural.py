#!/usr/bin/python

import math

weights = [1.232434,-3.754862,-1.342545]

input_x = [[1, 1, 0],[0,0,1]]
y = [0,1]


#s = w1p1+w2p2+w3p3+ ......

def add(weights,inputs):
 if len(weights) == len(inputs):
  s = 0
  for i in range(0, len(weights)):
   try :
    result = weights[i] * inputs[i]
    s += result
   except:
    return 0 
  return s
 else:
  print('Not match error.')
  exit()

def func_step(inputs):
 if inputs < 0 :
  return 0
 if inputs == 0 :
  return 0
 if inputs > 0 :
  return 1

#Wang wenjun
def func_sgn(inputs):
 if a < 0:
  return -1
 else:
  return 1

def func_linear(inputs):
 return inputs

#Liang renfeng
def func_ramp(inputs):
 if inputs < 0:
  return 0
 if inputs >= 0 and inputs <= 1:
  return inputs
 else:
  return 1

#Zhao peilian
def func_sigmoid(inputs):
    f=0
    demo=(1+math.exp(-1.0*inputs))
    if(demo==0):
      print('Not result')
    else:
      f=1.0/demo
    return f

#Wang wenjun
def func_Tanh(inputs):
 return (math.exp(inputs)-math.exp(-inputs))/(math.exp(inputs)+math.exp(-inputs))

def func_bias(inputs,b):
 return inputs + b * 1

def neural_output(weights,outputs):
 s = add(weights,outputs)
 b = func_bias(s, 1)
 outputs = func_sigmoid(b)
 return outputs

def loss(outputs, y, weights, bias, inputs):
 weights_new = []
 bias_new = []
 e = y - outputs
 for i in range(0, len(inputs)):
  weights[i] = weights[i] + e * inputs[i]
  weights_new.append(weights[i])
 bias_new = bias + e
 return weights_new,bias_new

def run_an_loop():
 weights = [1.232434,-3.754862,-1.342545]
 inputs = [1, -1, 0]
 bias = 1
 outputs = neural_output(weights, inputs)
 weights,bias = loss(outputs, y, weights, bias, inputs)
 print(weights, bias)
#print(neural_output([3.1215926,1.234567],[0,1]))

def train_for_1000steps():
 weights = [1.232434,-3.754862,-1.342545]
 inputs = [1, -1, 0]
 bias = 1
 for i in range(0, 1000):
  outputs = neural_output(weights, inputs)
  print(outputs)
  weights,bias = loss(outputs, y, weights, bias, inputs)
 print(weights)
 print(inputs)

#calculate single e
def loss_e(outputs, y):
  e = y - outputs
  return e

def train_on_batch():
 bias = 1
 e_avg = 0
 e_sum = 0
 for i in range(0,len(y)):
  outputs = neural_output(weights[i], input_x[i])
  e = loss_e(outputs,y[i])
  e_sum += e
 e_avg = e_sum/len(y)
 weight1,bias1 = update_hp(weights[0], bias)
 weight2,bias2 = update_hp(weights[1], bias)
 weight = avg_weight(weight1,weight2)
 bias = avg_bias(bias1,bias2)

 train_on_batch()
 
  
