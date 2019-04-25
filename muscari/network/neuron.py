#!/bin/python

'''
Implements a single Neuron.
I = Input
O = Output
W = Weight
B = Bias
A = Activation

            |\    +-----------+      +------------+     +----------+
I1 ----W1---| \   |  Weighted |      |            |     |          |
I2 ----W2---|  \  |    Sum    |---\  | Activation |---\ |  Output  |
...         |  /  |Sigma Ii*Wi|---/  |            |---/ |          |
In ----Wn---| /   |+ Bias (B) |      |            |     |          |
            |/    +-----------+      +------------+     +----------+
                      ^
                      |
B --------------------+
'''

import numpy as np

class Neuron:
  def __init__(self, activation, bias=0.0, initial_weights=None):
    self.b = bias
    self.a = activation
    self.w = initial_weights

  def output(self, inputs):
    if self.w is None:
      self.w = np.random.rand(inputs.shape[0])
    weighted_sum = np.sum(inputs * self.w) + self.b
    self.inputs = inputs
    self.o = self.a.fx(weighted_sum)
    return self.o

  def sigma(self, target= None):
    if target is None:
      return self.a.dfx(self.o)
    else:
      return (target - self.o)*self.a.dfx(self.o)

  def update_weights(self, new_weights):
    self.w = new_weights
