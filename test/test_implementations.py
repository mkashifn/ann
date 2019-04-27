#!/usr/bin/Python
import unittest
from muscari.network.neuron import Neuron
from muscari.network.layer import Layer
from muscari.network.network import Network
from muscari.math.functions import sigmoid
from muscari.math.estimators import mse
from muscari.network.sequential import Sequential
import numpy as np
#python -m unittest discover
class TestImplementations(unittest.TestCase):
  def setUp(self):
    pass
  def test_implementations_1(self):
    def print_output(headline, nn, ss, inputs):
      print headline
      print "  NN: ", nn.output(inputs)
      print "  SS: ", ss.output(inputs)

    def test_output(nn, ss, inputs, targets):
      print_output("No back propagation:", nn, ss, inputs)
      epoch = 10000
      for i in range(epoch):
        nn.output(inputs)
        nn.propagate_back(targets)
        ss.output(inputs)
        ss.propagate_back(targets)
      print_output("{} back propagation:".format(i), nn, ss, inputs)
      nno = nn.output(inputs)
      sso = ss.output(inputs)
      assert np.sum(nno - sso) == 0.0

    # *************
    # https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    # *************

    # Common Input
    inputs = np.array([[0.05, 0.10]], dtype=float)
    outputs = np.array([[0.01, 0.99]], dtype=float)
    weights = np.array([[0.15, 0.20], [0.25, 0.30]], dtype=float)

    # Test whole network
    nn = Network(mse, 0.5)
    ss = Sequential(mse, 0.5)
    nn.add_layer(2, sigmoid, 0.35, weights)
    ss.add_layer(2, sigmoid, 0.35, weights.T, 2)
    weights = np.array([[0.40, 0.45], [0.50, 0.55]], dtype=float)
    nn.add_layer(2, sigmoid, 0.60, weights)
    ss.add_layer(2, sigmoid, 0.60, weights.T)
    test_output(nn, ss, inputs, outputs)