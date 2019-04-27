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
    if self.layer_input_size is None and input_size is None:
      raise ValueError('input_size needs to be defined for the very first layer')
    if self.layer_input_size is not None and input_size is not None:
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
    self.layers.append(Layer(activation, bias, initial_weights))
    self.layer_input_size = neuron_count

  def feed_forward(self, input):
    output = None
    for l in self.layers:
      output = l.output(input)
      input = output
    self.o = output
    return self.o
  
  def output(self, input):
    return self.feed_forward(input)

  def update_layer_weights(self, layer, sigma, eta):
    d_weights = np.dot(layer.i.T, sigma) * eta
    new_weights = layer.w - d_weights
    old_weights = layer.w
    layer.w = new_weights
    return old_weights

  def propagate_back(self, targets):
    layers = self.layers[::-1] # reverse, need to start from output layer
    layer = layers[0]
    sigma = -self.loss.dfx(targets,layer.o)*layer.a.dfx(layer.o)
    old_weights = self.update_layer_weights(layer, sigma, self.eta)
    layers = layers[1:] # other layers
    for layer in layers:
      sigma = np.dot(sigma, old_weights.T * layer.a.dfx(layer.o))
      old_weights = self.update_layer_weights(layer, sigma, self.eta)
    