import sys

CUDA_ENABLED = True
try:
  import cupy as cp
except ImportError:
  import numpy as cp
  CUDA_ENABLED = False

import simulation

class CudaSimulation(simulation.FastSimulation):
  def __init__(self, kernel_type, initial_concentration, dt, lmbda, alpha):
    if not CUDA_ENABLED:
      print('CuPy is not installed! Falling back to numpy.', file=sys.stderr)
    initial_concentration = cp.array(initial_concentration)
    super().__init__(kernel_type, initial_concentration, dt, lmbda, alpha)

    self.V = cp.array(self.V)
    self.U = cp.array(self.U)
    self.Vj = cp.array(self.Vj)


  def K_nn(self, concentration):
    """
    sum_{i + j = k} K_{ij} * n_i * n_j
    """
    result = cp.zeros(len(concentration))
    for d in range(len(self.V)):
      first = self.V[d] * concentration
      second = self.U[:, d] * concentration
      result += self.convolve(first, second)[:len(concentration)]
    return result


  def convolve(self, first, second):
    shape = first.shape[0] + second.shape[0] - 1
    best_shape = int(2**cp.ceil(cp.log2(shape)))

    first_f = cp.fft.rfft(first, best_shape)
    second_f = cp.fft.rfft(second, best_shape)
    return cp.fft.irfft(first_f * second_f, best_shape)[:shape]
