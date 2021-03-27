import os
import time

import numpy as np
import tqdm

import simulation

import simulation_cuda as simcp

DATA_DIRECTORY = 'data'

class Experiment(object):
  def __init__(self, kernel_type, simulation_type, dt, lmbda, alpha, num_equations=None, initial=None,
      final_lambda=None, lambda_decay_type=None):
    if initial is None and num_equations is None:
      raise ValueError("At least one of num_equations and initial should be provided.")

    if initial is None:
      initial = np.zeros(num_equations + 1, dtype=np.float64)
      initial[1] = 1.0

    self.num_equations = len(initial)

    self.lmbda = lmbda
    self.final_lambda = final_lambda
    self.lambda_decay_type  = lambda_decay_type

    if simulation_type == "cuda":
      self.sim = simcp.CudaSimulation(kernel_type, initial, dt, lmbda, alpha)
    elif simulation_type == "fast":
      self.sim = simulation.FastSimulation(kernel_type, initial, dt, lmbda, alpha)
    elif simulation_type == "naive":
      self.sim = simulation.NaiveSimulation(kernel_type, initial, dt, lmbda, alpha)
    elif simulation_type == "backward_euler":
      self.sim = simulation.BackwardEulerSimulation(kernel_type, initial, dt, lmbda, alpha)
    elif simulation_type == "crank_nicolson":
      self.sim = simulation.CrankNicolsonSimulation(kernel_type, initial, dt, lmbda, alpha)
    else:
      raise ValueError("Unknown simulation type: " + str(simulation_type))


  def run_experiment(self, name, num_iters, checkpoint_freq):
    result_dir = os.path.join(DATA_DIRECTORY, name)
    solutions_dir = os.path.join(result_dir, 'solutions')
    os.makedirs(solutions_dir)

    lambdas = self.precompute_lambdas(num_iters)
    lambda_history = []
    time_history = []
    start_time = time.time()
    for iter in tqdm.tqdm(range(num_iters)):
      l = lambdas[iter]
      self.sim.update_lambda(l)
      self.sim.simulation_step()

      if iter % checkpoint_freq == 0:
        solutions_path = os.path.join(solutions_dir, str(iter))
        np.save(solutions_path, self.sim.concentration)
        lambda_history.append(l)
        time_history.append(time.time() - start_time)

    if num_iters % checkpoint_freq != 0:
      solutions_path = os.path.join(solutions_dir, str(num_iters - 1))
      np.save(solutions_path, self.sim.concentration)
      lambda_history.append(l)
      time_history.append(time.time() - start_time)

    lambdas_path = os.path.join(result_dir, 'lambda')
    np.save(lambdas_path, np.array(lambda_history))

    time_path = os.path.join(result_dir, 'time')
    np.save(time_path, np.array(time_history))

  def precompute_lambdas(self, num_iters):
    if self.lambda_decay_type is None:
      return np.ones(num_iters) * self.lmbda

    ts = np.linspace(0, 1.0, num_iters)
    lambda_range = self.lmbda - self.final_lambda
    lambdas = None
    if self.lambda_decay_type == 'logistic':
      shifted = ts - 0.4
      lambdas = 1 - 1 / (1 + np.exp(-12*shifted))

    if self.lambda_decay_type in ['exponential', 'steep_exponential']:
      scale = 10 if self.lambda_decay_type == 'steep_exponential' else 5
      exps = np.exp(-scale*ts)
      height = np.max(exps) - np.min(exps)
      lambdas = (exps - np.min(exps)) / height

    if lambdas is None:
      raise ValueError(f'Lambda decay type {self.lambda_decay_type} is not supported.')

    return lambdas * lambda_range + self.final_lambda



  def analytical_solution(self):
    k = np.arange(self.num_equations)
    return self.lmbda /(np.sqrt(np.pi) * k**(3/2)) * np.exp(-self.lmbda**2 * k)
