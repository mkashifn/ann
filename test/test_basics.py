#!/usr/bin/Python
import unittest
import numpy as np
from muscari.network.neuron import Neuron
from muscari.network.layer import Layer
from muscari.network.network import Network
from muscari.math.functions import sigmoid
from muscari.math.estimators import mse

#python -m unittest discover

class LayerDefinition:
  def __init__(self, neuron_count, activation, bias, weights):
    self.neuron_count = neuron_count
    self.activation = activation
    self.bias = bias
    self.weights = weights

def verify_single_neuron(input, weights, activation, bias):
  n = Neuron(activation, bias, weights)
  output = n.output(input)
  expected = activation.fx(np.sum(input*weights) + bias)
  #print (output, expected)
  assert output == expected
  return (output, expected)

def verify_single_layer(input, neuron_count, weights, activation, bias):
  l = Layer(neuron_count, activation, bias, weights)
  output = l.output(input)
  expected = []
  for i in range(neuron_count):
    (neuron_output, _) = verify_single_neuron(input, weights[i,:], activation, bias)
    expected.append(neuron_output)
  expected = np.array([expected], dtype=float)
  #print (output, expected)
  assert np.sum(output - expected) == 0.0
  return (output, expected)

def verify_whole_network(input, loss, eta, layer_defs, draw=False):
  expected = None
  layer_input = input
  n = Network(loss, eta)
  for l in layer_defs:
    n.add_layer(l.neuron_count, l.activation, l.bias, l.weights)
    (layer_output, _) = verify_single_layer(layer_input, l.neuron_count, l.weights, l.activation, l.bias)
    expected = layer_input = layer_output
  output = n.output(input)
  #print (output, expected)
  assert np.sum(output - expected) == 0.0
  if draw:
    n.draw(input, expected, file="unittest_verify_whole_network", cleanup=True)
  return (output, expected)

class TestBasics(unittest.TestCase):
  def setUp(self):
    pass
  def test_neuron_1(self):
    '''Test single neuron.'''
    input = np.array([[0.05, 0.10]], dtype=float)
    weights = np.array([[0.15, 0.20]], dtype=float)
    bias = 0.35
    activation = sigmoid

    (output, expected) = verify_single_neuron(input, weights, activation, bias)

  def test_layer_1(self):
    '''Test single layer.'''
    input = np.array([[0.05, 0.10]], dtype=float)
    weights = np.array([[0.15, 0.20], [0.25, 0.30]], dtype=float)
    bias = 0.35
    activation = sigmoid
    neuron_count = weights.shape[0]

    (output, expected) = verify_single_layer(input, neuron_count, weights, activation, bias)

  def test_network_1(self):
    '''Test whole network.'''
    input = np.array([[0.05, 0.10]], dtype=float)
    layer_defs = []
    loss = mse
    eta = 0.5

    layer1_activation = sigmoid
    layer1_bias = 0.35
    layer1_weights = np.array([[0.15, 0.20], [0.25, 0.30]], dtype=float)
    layer1_neuron_count = layer1_weights.shape[0]
    layer1 = LayerDefinition(layer1_neuron_count, layer1_activation, layer1_bias, layer1_weights)
    layer_defs.append(layer1)

    layer2_activation = sigmoid
    layer2_bias = 0.60
    layer2_weights = np.array([[0.40, 0.45], [0.50, 0.55]], dtype=float)
    layer2_neuron_count = layer2_weights.shape[0]
    layer2 = LayerDefinition(layer2_neuron_count, layer2_activation, layer2_bias, layer2_weights)
    layer_defs.append(layer2)

    (output, expected) = verify_whole_network(input, loss, eta, layer_defs, draw=True)
