#!/bin/python
import numpy as np
from layer import Layer
from graphviz import Digraph
from graphviz import Graph

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
      #print("sum_sigma:", sum_sigma_weights)
      errors = neuron.inputs * sigma
      new_weights = np.subtract(neuron.w, self.eta * errors)
      #print ("Target:", targets[:,i], "Sigma:", sigma, "W:", neuron.w, "New Weights:", new_weights)
      neuron.update_weights(new_weights)
      i += 1
    layers = layers[1:] # other layers
    for layer in layers:
      new_sum_sigma_weights = 0
      i = 0
      for neuron in layer.neurons:
        sigma = sum_sigma_weights[:,i] * neuron.sigma()
        new_sum_sigma_weights += neuron.w * sigma
        errors = neuron.inputs * sigma
        new_weights = np.subtract(neuron.w, self.eta * errors)
        #print ("Sigma:", sigma, "W:", neuron.w, "New Weights:", new_weights)
        neuron.update_weights(new_weights)
        i += 1
      sum_sigma_weights = new_sum_sigma_weights

  def train(self, inputs, targets):
    A = targets
    B = self.feed_forward(inputs)
    print ("Loss:", self.loss(A, B))
    self.propagate_back(targets)
    #print ("Loss:", self.loss(A, self.feed_forward(inputs)))

  def draw(self, input, file="ann", dir="draw", view=True, cleanup=False):
    """
    dot = Digraph(comment='The Round Table')
    dot.node('A', 'King Arthur')
    dot.node('B', 'Sir Bedevere the Wise')
    dot.node('L', 'Sir Lancelot the Brave')
    dot.edges(['AB', 'AL'])
    dot.edge('B', 'L', constraint='false')
    dot.render(file, dir, view=view, cleanup=cleanup)
    """
    graph = Graph(directory='graphs', format='pdf',
                  graph_attr=dict(ranksep='2', rankdir='LR', color='white', splines='line'),
                  node_attr=dict(label='', shape='circle', width='0.1'))


    def draw_cluster(name, length, fillcolor="#FFFFFF", subscript=""):
      names = []
      with graph.subgraph(name='cluster_{name}'.format(name=name)) as c:
        c.attr(label=name)
        for i in range(length):
          name_str = '{name}_{i}'.format(name=name, i=i)
          label = '{id}{ss}-{i}'.format(ss=subscript,id=name[0],i=i)
          #label = "\$\sigma_1\$"
          #label = u'\u0220' #Unicode: https://unicode-table.com/en/#control-character
          c.node(name_str, label, color='black',style='filled',fillcolor=fillcolor)
          #c.node(name_str, r"$\sigma_1$")
          names.append(name_str)
      return names

    def draw_connections(src, dst):
      for s in src:
        for d in dst:
          graph.edge(s, d)

    src = draw_cluster('input', input.shape[1], "#FFFF00")
    i = 1
    for layer in self.layers[:-1]:
      dst = draw_cluster('hidden_layer_{i}'.format(i=i), layer.count, "#04ADFC", i)
      i += 1
      draw_connections(src, dst)
      src = dst

    layer = self.layers[-1]
    dst = draw_cluster('output', layer.count, "#00FF00")
    draw_connections(src, dst)
      
    """
    source_active = [0, 1, 2, 3]
    sink_active = [2, 3]

    for i_input in source_active:
        for i_output in sink_active:
            graph.edge('input_{i_input}'.format(i_input=i_input), 'output_{i_output}'.format(i_output=i_output))"""
    graph.render(file, dir, view=view, cleanup=cleanup)