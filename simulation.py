import numpy as np
from scipy.signal import fftconvolve

class Simulation(object):
  def __init__(self, num_equations, dt, lmbda, alpha):
    self.concentration = np.zeros(1 + num_equations, dtype=np.float64)
    self.concentration[1] = 1.0
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
    sum_{i, j >= 2} K_{ij} * (i + j) * n_i * n_j -> float
    """
    raise NotImplementedError

  def K_n(self, concentration):
    """
    sum_{j >= 1} K_{k, j} * n_{j} -> vector len(concentration)
    """
    raise NotImplementedError

  def j_K_1j_n(self, concentration):
    """
    sum_{j >= 2} j * K_{1j}*n_j -> float
    """
    raise NotImplementedError

  def compute_update(self, concentration):
    update = np.zeros_like(concentration)
    K_n = self.K_n(concentration)
    update[1] = (-concentration[1] * K_n[1] + self.lmbda / 2 * self.K_ij_nn(concentration)
        + self.lmbda * concentration[1] * self.j_K_1j_n(concentration))
    update[2:] = (0.5 * self.K_nn(concentration) - (1 + self.lmbda) * concentration * K_n)[2:]
    return update

  def simulation_step(self):
    update = self.compute_update(self.concentration)
    self.concentration += self.dt * update

  def run_simulation(self, num_iters):
    for i in range(num_iters):
      # print(i)
      self.simulation_step()
    return self.concentration


class NaiveSimulation(Simulation):
  def __init__(self, num_equations, dt, lmbda, alpha):
    super().__init__(num_equations, dt, lmbda, alpha)

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
    sum_{i, j >= 2} K_{ij} * (i + j) * n_i * n_j
    """
    result = 0
    for i in range(2, len(concentration)):
      for j in range(2, len(concentration)):
        result += ((i / j)**self.alpha + (j / i)**self.alpha) * (i + j) * concentration[i] * concentration[j]
    return result

  def K_n(self, concentration):
    """
    sum_{j >= 1} K_{k, j} * n_{j} -> vector
    """
    result = np.zeros_like(concentration)

    iss = np.arange(len(concentration)).reshape((-1, 1))
    js = np.arange(len(concentration))
    K = (iss / js)**self.alpha + (js / iss)**self.alpha
    K[~np.isfinite(K)] = 0
    return K @ concentration

  def j_K_1j_n(self, concentration):
    """
    sum_{j >= 2} j * K_{1j}*n_j
    """
    js = np.arange(2, len(concentration))
    return js @ ((1/js**self.alpha + js**self.alpha) * concentration[2:])


class FastSimulation(Simulation):
  def __init__(self, num_equations, dt, lmbda, alpha):
    super().__init__(num_equations, dt, lmbda, alpha)

    self.trunc_js = np.arange(2, num_equations + 1, dtype=np.float64)
    js = self.trunc_js
    self.j_K_1j = js * (1/js**self.alpha + js**self.alpha)

    js = np.arange(1, num_equations + 1, dtype=np.float64)
    self.V = np.zeros((2, num_equations + 1))
    self.V[0, 1:] = js**(-self.alpha)
    self.V[1, 1:] = js**(self.alpha)
    self.U = self.V.T[:, ::-1]

    self.trunc_Vj = self.V[:, 2:] * self.trunc_js[None, :]
    self.trunc_U = self.U[2:, :]

  def K_nn(self, concentration):
    """
    sum_{i + j = k} K_{ij} * n_i * n_j
    """
    first = self.V[0] * concentration
    second = self.V[1] * concentration
    return 2 * fftconvolve(first, second)[:len(concentration)]


  def K_ij_nn(self, concentration):
    """
    sum_{i, j >= 2} K_{ij} * (i + j) * n_i * n_j
    """
    trunc_concentration = concentration[2:]
    right = self.trunc_Vj @ (trunc_concentration)
    left = trunc_concentration @ self.trunc_U
    return 2 * left @ right


  def K_n(self, concentration):
    """
    sum_{j >= 1} K_{k, j} * n_{j} -> vector
    """
    return self.U @ (self.V @ concentration)


  def j_K_1j_n(self, concentration):
    """
    sum_{j >= 2} j * K_{1j}*n_j
    """
    return self.j_K_1j @ concentration[2:]


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
