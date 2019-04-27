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
    self.count = weights.shape[1] #number of neurons
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
    self.layer_count = 0
    self.weight_count = 1

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
      x = expected_weights_shape[0]
      y = expected_weights_shape[1]
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

  def train(self, inputs, targets, epochs):
    for i in range(epochs):
      A = targets
      B = self.feed_forward(inputs)
      self.propagate_back(np.array(targets))
      if (i%100) == 0:
        print ("Epoch: {i}, Loss: {loss}".format(i=i, loss = self.loss(A, B)))

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
      sigma = np.dot(sigma, old_weights.T) * layer.a.dfx(layer.o)
      old_weights = self.update_layer_weights(layer, sigma, self.eta)

  def draw(self, inputs, targets, file="sequential", dir="draw", view=True, cleanup=False):
    graph = Digraph(directory='graphs', format='pdf',
                  graph_attr=dict(ranksep='2', rankdir='LR', color='white', splines='line'),
                  node_attr=dict(label='', shape='circle', width='0.1'))
    graph.attr(label='Hiw')
    np_formatter = {'float_kind':lambda x: "%.9g" % x}

    def increment_glc():
      self.layer_count += 1

    def increment_gwc():
      self.weight_count += 1

    def draw_cluster(name, length, values, fillcolor="#FFFFFF", subscript="", targets=None):
      names = []
      with graph.subgraph(name='cluster_{name}'.format(name=name)) as c:
        c.attr(label='{name}\n(layer {glc})'.format(name=name, glc = self.layer_count))
        increment_glc()

        for i in range(length):
          name_str = '{name}_{i}'.format(name=name, i=i)
          outcome = values[:,i]
          label = '{id}{ss}-{i}\n= {val}'.format(ss=subscript,id=name[0],i=i, val='{}'.format(np.array2string(outcome, formatter=np_formatter)))
          if targets is not None:
            target = targets[:,i]
            loss = self.loss(target, outcome) / targets.shape[1]
            label += '\ntarget: {}'.format(np.array2string(target, formatter=np_formatter))
            label += '\nloss: {}'.format(np.array2string(loss, formatter=np_formatter))
          #label = "\$\sigma_1\$"
          #label = u'\u0220' #Unicode: https://unicode-table.com/en/#control-character
          c.node(name_str, label, color='black',style='filled',fillcolor=fillcolor)
          #c.node(name_str, r"$\sigma_1$")
          names.append(name_str)
      return names

    def draw_connections(src, dst, dst_layer):
      auto_colors = ['#00A2EB','#4BC000','#9966FF','#FF7700','#808000','#800000','#0020F0','#00F0F0','#8463FF','#E6194B']
      w = dst_layer.w.flatten('F')

      i = 0;
      for d in dst:
        for s in src:
          color = auto_colors[i % len(auto_colors)]
          graph.edge(s, d, fontcolor=color, color=color, label='w{wl} = {weight}'.format(wl=self.weight_count, weight=w[i]))
          increment_gwc()
          i += 1

    src = draw_cluster('input', inputs.shape[1], inputs, "#FFFF00")
    i = 1
    layer_input = inputs
    for layer in self.layers[:-1]:
      layer_output = layer.output(layer_input)
      dst = draw_cluster('hidden_layer_{i}'.format(i=i), layer.count, layer_output, "#04ADFC", i)
      i += 1
      draw_connections(src, dst, layer)
      src = dst
      layer_input = layer_output

    layer = self.layers[-1]
    dst = draw_cluster('output', layer.count, layer.output(layer_input), "#00FF00", targets=targets)
    draw_connections(src, dst, layer)
    A = targets
    B = self.feed_forward(inputs)
    tl = self.loss(A, B)
    tl = np.array2string(tl, formatter=np_formatter)
    graph.attr(label='{file}, total loss = {tl}'.format(file=file, tl=tl))

    graph.render(file, dir, view=view, cleanup=cleanup)