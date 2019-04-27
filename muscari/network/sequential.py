#!/bin/python
import numpy as np
from graphviz import Digraph
from graphviz import Graph
import warnings

class Layer:
  def __init__(self, activation, bias, weights):
    self.a = activation
    self.b = bias
    self.w = weights
  def output(self, input):
    self.i = input
    self.o = self.a.fx(np.dot(self.i, self.w) + self.b)
    return self.o

class Sequential:
  def __init__(self, loss, eta):
    self.layers = []
    self.loss = loss
    self.eta = eta # learning rate
    self.layer_input_size = None #number of inputs for the NEXT layer

    self.layers = []

  def add_layer(self, neuron_count, activation, bias, initial_weights=None, input_size=None):
    # Validate the inputs
    if self.next_layer_input is None and input_size is None:
      raise ValueError('input_size needs to be defined for the very first layer')
    if self.next_layer_input is not None and input_size is not None:
      raise ValueError('input_size is not required for this layer')
    if input_size is not None:
      layer_input_size = input_size
    else:
      layer_input_size = self.layer_input_size
    expected_weights_shape = (layer_input_size, neuron_count)
    if initial_weights is None:
      x = np.random.rand[0]
      y = np.random.rand[1]
      initial_weights = np.random.rand(x,y)
    if initial_weights.shape != expected_weights_shape:
      warnings.warn("wights matrix is not properly formed, expected {e} vs actual {a}".format(e =expected_weights_shape, a=initial_weights.shape), UserWarning)
      initial_weights.shape = expected_weights_shape
    self.layers.append(Layer(activation, bias, weights))

  def feed_forward(self, input):
    output = None
    for l in self.layers:
      output = l.output(input)
      input = output
    self.o = output
    return self.o