import sys

CUDA_ENABLED = True
try:
  import cupy as cp
except ImportError:
  import numpy as cp
  CUDA_ENABLED = False

import simulation

class CudaSimulation(simulation.FastSimulation):
  def __init__(self, initial_concentration, dt, lmbda, alpha):
    if not CUDA_ENABLED:
      print('CuPy is not installed! Falling back to numpy.', file=sys.stderr)
    initial_concentration = cp.array(initial_concentration)
    super().__init__(initial_concentration, dt, lmbda, alpha)


    js = cp.arange(0, len(self.concentration), dtype=cp.float64)
    self.V = cp.zeros((2, len(self.concentration)))
    self.V[0, 1:] = js[1:]**(-self.alpha)
    self.V[1, 1:] = js[1:]**(self.alpha)
    self.U = self.V.T[:, ::-1]

    self.Vj = self.V * js[None, :]


  def K_nn(self, concentration):
    """
    sum_{i + j = k} K_{ij} * n_i * n_j
    """
    first = self.V[0] * concentration
    second = self.V[1] * concentration
    return 2 * self.convolve(first, second)[:len(concentration)]

  def convolve(self, first, second):
    shape = first.shape[0] + second.shape[0] - 1
    best_shape = int(2**cp.ceil(cp.log2(shape)))

    first_f = cp.fft.rfft(first, best_shape)
    second_f = cp.fft.rfft(second, best_shape)
    return cp.fft.irfft(first_f * second_f, best_shape)[:shape]
