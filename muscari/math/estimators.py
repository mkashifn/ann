#!/bin/python
import numpy as np

class Estimator:
  def __init__(self):
    pass
  def __call__(self, A, B):
    return 0

class MSE(Estimator):
  def __init__(self):
    pass
  def __call__(self, A, B):
    #print ("A:", A, "B:", B, "Diff:", np.subtract(A, B), "SqDiff:", np.square(np.subtract(A, B)), "MSE:", np.square(np.subtract(A, B)).mean())
    return np.square(np.subtract(A, B)).mean()

mse = MSE()