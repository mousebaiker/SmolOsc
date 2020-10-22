import numpy as np
from scipy.signal import fftconvolve

import kernels

class Simulation(object):
  def __init__(self, initial_concentration, dt, lmbda, alpha):
    self.concentration = np.copy(initial_concentration)
    self.dt = dt
    self.lmbda = lmbda
    self.alpha = alpha

  def K_nn(self, concentration):
    """
    sum_{i + j = k} K_{ij} * n_i * n_j  -> vector len(concentration)
    """
    raise NotImplementedError

  def K_ij_nn(self, concentration):
    """
    sum_{i, j >= 1} K_{ij} * (i + j) * n_i * n_j -> float
    """
    raise NotImplementedError

  def K_n(self, concentration):
    """
    sum_{j >= 1} K_{k, j} * n_{j} -> vector len(concentration)
    """
    raise NotImplementedError


  def update_lambda(self, l):
    self.lmbda = l

  def compute_update(self, concentration):
    update = np.zeros_like(concentration)
    K_n = self.K_n(concentration)
    update[1] = self.lmbda / 2 * self.K_ij_nn(concentration)
    update[2:] = 0.5 * self.K_nn(concentration)[2:]
    update -= (1 + self.lmbda) * concentration * K_n
    return update

  def simulation_step(self):
    update = self.compute_update(self.concentration)
    self.concentration += self.dt * update

  def run_simulation(self, num_iters):
    for _ in range(num_iters):
      self.simulation_step()
    return self.concentration


class NaiveSimulation(Simulation):
  def __init__(self, kernel_type, initial_concentration, dt, lmbda, alpha):
    super().__init__(initial_concentration, dt, lmbda, alpha)
    self.kernel_type = kernels.KERNELS[kernel_type]

  def K_nn(self, concentration):
    """
    sum_{i + j = k} K_{ij} * n_i * n_j
    """
    result = np.zeros_like(concentration)
    for k in range(2, len(concentration)):
      for i in range(1, k):
        K_ij = self.kernel_type(i, k-i, alpha=self.alpha)
        result[k] += concentration[i] * concentration[k-i] * K_ij
    return result

  def K_ij_nn(self, concentration):
    """
    sum_{i, j >= 1} K_{ij} * (i + j) * n_i * n_j
    """
    result = 0
    for i in range(1, len(concentration)):
      for j in range(1, len(concentration)):
        K_ij = self.kernel_type(i, j, alpha=self.alpha)
        result += K_ij * (i + j) * concentration[i] * concentration[j]
    return result

  def K_n(self, concentration):
    """
    sum_{j >= 1} K_{k, j} * n_{j} -> vector
    """
    iss = np.arange(len(concentration)).reshape((-1, 1))
    js = np.arange(len(concentration))
    K = self.kernel_type(iss, js, alpha=self.alpha)
    K[~np.isfinite(K)] = 0
    return K @ concentration


class FastSimulation(Simulation):
  def __init__(self, kernel_type, initial_concentration, dt, lmbda, alpha):
    super().__init__(initial_concentration, dt, lmbda, alpha)
    self.kernel_type = kernel_type

    js = np.arange(0, len(self.concentration), dtype=np.float64)
    self.init_UV(kernel_type, len(self.concentration))

    self.Vj = self.V * js[None, :]

  def init_UV(self, kernel_type, num_equations):
    if kernel_type == 'constant':
      self.V = np.ones((1, num_equations))
      self.V[0, 0] = 0.0
      self.U = self.V.T
    elif kernel_type == 'brownian':
      js = np.arange(0, num_equations, dtype=np.float64)
      self.V = np.zeros((2, num_equations))
      self.V[0, 1:] = js[1:]**(-self.alpha)
      self.V[1, 1:] = js[1:]**(self.alpha)
      self.U = self.V.T[:, ::-1]
    elif kernel_type == 'ballistic':
      i = np.arange(0, num_equations)
      j = np.arange(0, num_equations).reshape((-1, 1))
      K = (i**(1/3.0) + j**(1/3.0))**2.0 * (1.0 / i + 1.0 / j)**0.5
      K[~np.isfinite(K)] = 0.0
      u, s, v = np.linalg.svd(K)
      rank = np.sum(s > 1e-10)
      self.V = v[:rank, :]
      self.U = u[:, :rank] * s[:rank]


  def K_nn(self, concentration):
    """
    sum_{i + j = k} K_{ij} * n_i * n_j
    """
    result = np.zeros(len(concentration))
    for d in range(len(self.V)):
      first = self.V[d] * concentration
      second = self.U[:, d] * concentration
      result += fftconvolve(first, second)[:len(concentration)]
    return result


  def K_ij_nn(self, concentration):
    """
    sum_{i, j >= 1} K_{ij} * (i + j) * n_i * n_j
    """
    right = self.Vj @ concentration
    left = concentration @ self.U
    return 2 * left @ right


  def K_n(self, concentration):
    """
    sum_{j >= 1} K_{k, j} * n_{j} -> vector
    """
    return self.U @ (self.V @ concentration)


class ConstantSourceSimulation(FastSimulation):
  def __init__(self, num_equations, dt, alpha):
    lmbda = 0
    super().__init__(num_equations, dt, lmbda, alpha)

  def compute_update(self, concentration):
    update = np.zeros_like(concentration)
    K_n = self.K_n(concentration)
    update[1] = (-concentration[1] * K_n[1] + 1)
    update[2:] = (0.5 * self.K_nn(concentration) - (1 + self.lmbda) * concentration * K_n)[2:]
    return update
