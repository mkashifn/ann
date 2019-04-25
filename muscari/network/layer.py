#!/bin/python

# +++++++++++++++++++++++++++++++++++++++++
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
#    https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py
# +++++++++++++++++++++++++++++++++++++++++

import numpy as np
from neuron import Neuron

class Layer:
  def __init__(self, count, activation, bias=0.0, initial_weights = None):
    self.neurons = []
    self.count = count
    for i in range(count):
      neuron_weight = np.array([initial_weights[i,:]])
      self.neurons.append(Neuron(activation, bias, neuron_weight))

  def output(self, inputs):
    output = []
    for neuron in self.neurons:
      output.append(neuron.output(inputs))
    return np.array([output], dtype=float)