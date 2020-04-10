import os

import numpy as np
import tqdm

import simulation

DATA_DIRECTORY = 'data'

class Experiment(object):
  def __init__(self, dt, lmbda, alpha, num_equations=None, initial=None):
    if initial is None and num_equations is None:
      raise ValueError("At least one of num_equations and initial should be provided.")

    if initial is None:
      initial = np.zeros(num_equations + 1, dtype=np.float64)
      initial[1] = 1.0

    self.num_equations = len(initial)

    self.lmbda = lmbda
    self.sim = simulation.FastSimulation(initial, dt, lmbda, alpha)

  def run_experiment(self, name, num_iters, checkpoint_freq):
    result_dir = os.path.join(DATA_DIRECTORY, name)
    solutions_dir = os.path.join(result_dir, 'solutions')
    os.makedirs(solutions_dir)

    lambdas = []
    for iter in tqdm.tqdm(range(num_iters)):
      l = self.compute_lambda(iter)
      self.sim.update_lambda(l)
      self.sim.simulation_step()

      if iter % checkpoint_freq == 0:
        solutions_path = os.path.join(solutions_dir, str(iter))
        np.save(solutions_path, self.sim.concentration)
        lambdas.append(l)

    if num_iters % checkpoint_freq != 0:
      solutions_path = os.path.join(solutions_dir, str(num_iters - 1))
      np.save(solutions_path, self.sim.concentration)
      lambdas.append(l)

    params_path = os.path.join(result_dir, 'params')
    np.save(params_path, np.array(lambdas))

  def compute_lambda(self, iter):
    return self.lmbda

  def analytical_solution(self):
    k = np.arange(self.num_equations)
    return self.lmbda /(np.sqrt(np.pi) * k**(3/2)) * np.exp(-self.lmbda**2 * k)







