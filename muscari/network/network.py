#!/bin/python
import numpy as np
from layer import Layer

class Network:
  def __init__(self, loss, eta):
    self.layers = []
    self.loss = loss
    self.eta = eta # learning rate

  def add_layer(self, count, activation, bias=0.0, initial_weights=None):
    self.layers.append(Layer(count, activation, bias, initial_weights))

  def output(self, inputs):
    return self.feed_forward(inputs)

  def feed_forward(self, inputs):
    c_inputs = inputs
    c_output = []
    for layer in self.layers:
      c_output = layer.output(c_inputs)
      c_inputs = c_output
    return c_output

  def propagate_back(self, targets):
    layers = self.layers[::-1] # reverse, need to start from output layer
    # computations for output layers
    i = 0
    sum_sigma_weights = 0
    for neuron in layers[0].neurons:
      sigma = -1 * neuron.sigma(targets[:,i])
      sum_sigma_weights += neuron.w * sigma
      errors = neuron.inputs.T * sigma
      new_weights = np.subtract(neuron.w, self.eta * errors)
      print ("Target:", targets[:,i], "Sigma:", sigma, "W:", neuron.w, "New Weights:", new_weights)
      neuron.update_weights(new_weights)
      i += 1
    layers = layers[1:] # other layers
    for layer in layers:
      new_sum_sigma_weights = 0
      i = 0
      for neuron in layer.neurons:
        sigma = sum_sigma_weights[i] * neuron.sigma()
        new_sum_sigma_weights += neuron.w * sigma
        errors = np.dot(neuron.inputs.T, sigma)
        new_weights = np.subtract(neuron.w, self.eta * errors)
        print ("Sigma:", sigma, "W:", neuron.w, "New Weights:", new_weights)
        neuron.update_weights(new_weights)
        i += 1
      sum_sigma_weights = new_sum_sigma_weights

  def train(self, inputs, targets):
    A = targets
    B = self.feed_forward(inputs)
    self.propagate_back(targets)
    return ("Loss:", self.loss(A, B))