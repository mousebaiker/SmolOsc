import numpy as np
from scipy.signal import fftconvolve

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
  def __init__(self, initial_concentration, dt, lmbda, alpha):
    super().__init__(initial_concentration, dt, lmbda, alpha)

  def K_nn(self, concentration):
    """
    sum_{i + j = k} K_{ij} * n_i * n_j
    """
    result = np.zeros_like(concentration)
    for k in range(2, len(concentration)):
      for i in range(1, k):
        result[k] += concentration[i] * concentration[k-i] * ((i / (k - i))**self.alpha + ((k - i) / i)**self.alpha)
    return result

  def K_ij_nn(self, concentration):
    """
    sum_{i, j >= 1} K_{ij} * (i + j) * n_i * n_j
    """
    result = 0
    for i in range(1, len(concentration)):
      for j in range(1, len(concentration)):
        result += ((i / j)**self.alpha + (j / i)**self.alpha) * (i + j) * concentration[i] * concentration[j]
    return result

  def K_n(self, concentration):
    """
    sum_{j >= 1} K_{k, j} * n_{j} -> vector
    """
    iss = np.arange(len(concentration)).reshape((-1, 1))
    js = np.arange(len(concentration))
    K = (iss / js)**self.alpha + (js / iss)**self.alpha
    K[~np.isfinite(K)] = 0
    return K @ concentration


class FastSimulation(Simulation):
  def __init__(self, initial_concentration, dt, lmbda, alpha):
    super().__init__(initial_concentration, dt, lmbda, alpha)

    js = np.arange(0, len(self.concentration), dtype=np.float64)
    self.V = np.zeros((2, len(self.concentration)))
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
    return 2 * fftconvolve(first, second)[:len(concentration)]


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
