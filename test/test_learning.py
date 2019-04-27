#!/usr/bin/Python
import unittest
import numpy as np
from muscari.network.neuron import Neuron
from muscari.network.layer import Layer
from muscari.network.network import Network
from muscari.network.sequential import Sequential
from muscari.math.functions import sigmoid
from muscari.math.estimators import mse
from utils.disk import save_object, load_object

#python -m unittest discover
class TestLearning(unittest.TestCase):
  def setUp(self):
    pass
  def test_learning_1(self):
    """Test learning after 10000 epochs.
       https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    """
    # Common Input
    inputs = np.array([[0.05, 0.10]], dtype=float)
    outputs = np.array([[0.01, 0.99]], dtype=float)
    weights = np.array([[0.15, 0.20], [0.25, 0.30]], dtype=float)
    neuron_count = weights.shape[0]

    # Test whole network
    nn = Network(mse, 0.5)
    nn.add_layer(neuron_count, sigmoid, 0.35, weights)

    weights = np.array([[0.40, 0.45], [0.50, 0.55]], dtype=float)
    neuron_count = weights.shape[0]
    nn.add_layer(neuron_count, sigmoid, 0.60, weights)
    print (nn.output(inputs))
    epochs = 10000
    nn.train(inputs, outputs, epochs)
    print (nn.output(inputs))
    nn.draw(inputs, outputs, file="unittest_test_learning_1", cleanup=True)
    assert True == True
    
  def test_learning_2(self):
    """Test learning after 1000 epochs.
       https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
    """
    inputs = np.array([[0,0,1],
                      [0,1,1],
                      [1,0,1],
                      [1,1,1]])
    outputs = np.array([[0],[1],[1],[0]])
    weights = None
    nn = Network(mse, 2)
    nn.add_layer(5, sigmoid, 0.0, weights)
    nn.add_layer(16, sigmoid, 0.0, weights)
    nn.add_layer(3, sigmoid, 0.0, weights)
    nn.add_layer(1, sigmoid, 0.0, weights)
    #nn.draw(inputs, outputs)
    #nn.draw(np.array([inputs[0, :]]), np.array([outputs[0,:]]))
    #exit(0)
    #print (nn.output(np.array([inputs[0,:]])))
    epochs = 1000
    #for i in range(epochs):
    nn.train(inputs, outputs, epochs)
    print (nn.output(inputs[0,:]))
    print (nn.output(inputs[1,:]))
    print (nn.output(inputs[2,:]))
    print (nn.output(inputs[3,:]))
    #nn.draw(np.array([inputs[0, :]]), np.array([outputs[0,:]]))
    nn.draw(inputs, outputs, file="unittest_test_learning_2", cleanup=True)
    assert True == True

  def test_learning_3(self):
    """Test learning after 1500 epochs.
       https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
    """
    inputs = np.array([[0,0,1],
                      [0,1,1],
                      [1,0,1],
                      [1,1,1]])
    outputs = np.array([[0],[1],[1],[0]])
    weights = None
    nn = Sequential(mse, 1)
    nn.add_layer(4, sigmoid, 0.0, weights, 3)
    nn.add_layer(1, sigmoid, 0.0, weights)
    epochs = 1500
    nn.train(inputs, outputs, epochs)
    print (nn.output(inputs[0,:]))
    print (nn.output(inputs[1,:]))
    print (nn.output(inputs[2,:]))
    print (nn.output(inputs[3,:]))
    nn.draw(inputs, outputs, file="unittest_test_learning_3", cleanup=True)
    print "NN: ", nn.output(inputs)
    filename = 'draw/ut3-trained-nn.nn'
    save_object(nn, filename)
    del nn
    nnl = load_object(filename)
    print "NN1: ", nnl.output(inputs)
    nnl.draw(inputs, outputs, file="unittest_test_learning_3-loaded-nn", cleanup=True)
    assert True == True