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
    self.layer_count = 0
    self.weight_count = 1

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
      sigma = np.array([-1 * neuron.sigma(targets[:,i])])
      sum_sigma_weights += neuron.w * sigma
      #print("sum_sigma:", sum_sigma_weights)
      errors = neuron.inputs * sigma
      new_weights = np.subtract(neuron.w, self.eta * errors)
      #print ("Target:", targets[:,i], "Sigma:", sigma, "W:", neuron.w, "New Weights:", new_weights)
      neuron.update_weights(new_weights)
      i += 1
    #print("OUTPUT SIGMA: ", sum_sigma_weights)
    layers = layers[1:] # other layers
    for layer in layers:
      new_sum_sigma_weights = 0
      i = 0
      for neuron in layer.neurons:
        sigma = np.array([sum_sigma_weights[:,i]]) * neuron.sigma()
        new_sum_sigma_weights += neuron.w * sigma
        errors = neuron.inputs * sigma
        new_weights = np.subtract(neuron.w, self.eta * errors)
        #print ("Sigma:", sigma, "W:", neuron.w, "New Weights:", new_weights)
        neuron.update_weights(new_weights)
        i += 1
      sum_sigma_weights = new_sum_sigma_weights
      #print("LAYER SIGMA: ", sum_sigma_weights)

  def train(self, inputs, targets, epochs):
    batch_size = inputs.shape[0]
    for i in range(epochs):
      #A = targets
      #B = self.feed_forward(inputs)
      #self.propagate_back(np.array(targets))
      for b in range(batch_size):
        A = targets
        B = self.feed_forward(np.array([inputs[b,:]]))
        target = np.array([targets[b,:]])
        self.propagate_back(target)

      if (i%100) == 0:
        print ("Epoch:", i, "    Loss:", self.loss(A, B))
    print ("Loss:", self.loss(A, self.feed_forward(inputs)))

  def draw(self, inputs, targets, file="ann", dir="draw", view=True, cleanup=False):
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
      w = np.array([[]])
      for n in dst_layer.neurons:
        w = np.append(w, n.w)
      w = w.flatten()

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