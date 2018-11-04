#!/usr/bin/python

import math
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
def function_sigmoid(inputs):
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
