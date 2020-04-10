import argparse
import sys
import os

import yaml
import numpy as np
import matplotlib

import experiment
import plotting
import util

def load_required_param(config, key):
  param = config.get(key)
  if param is None:
    print(f'{key} param is required for experiment execution', file=sys.stderr)
  return param

def main():
  parser = argparse.ArgumentParser(description='Run experiment and save the derivative results.')
  parser.add_argument('yaml_config', help='Config YAML file specifying the parameters of experiment')

  args = parser.parse_args()
  with open(args.yaml_config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  dt = load_required_param(config, 'dt')
  lmbda = load_required_param(config, 'lambda')
  alpha = load_required_param(config, 'alpha')
  name = load_required_param(config, 'experiment_name')
  num_iters = load_required_param(config, 'num_iters')
  checkpoint_freq = load_required_param(config, 'checkpoint_frequency')


  if any(map(lambda x: x is None, [dt, lmbda, alpha, name, num_iters, checkpoint_freq])):
    sys.exit(1)

  num_equations = config.get('num_equations')
  path_to_initial = config.get('path_to_initial')
  if num_equations is None and path_to_initial is None:
    print(f'Please specify number of equations (num_equations key) or initial solution (path_to_initial key)', file=sys.stderr)
    sys.exit(1)

  initial = None
  if path_to_initial is not None:
    initial = np.load(path_to_initial)

  e = experiment.Experiment(dt=dt, lmbda=lmbda, alpha=alpha, num_equations=num_equations, initial=initial)

  print(f'Starting experiment {name}')
  e.run_experiment(name, num_iters, checkpoint_freq)

  print('Simulation finished. Loading results')
  iters, solutions = util.load_solutions(name)
  ts = iters * dt

  print('Computing moments')
  zeroth = util.compute_batch_moments(solutions, 0)
  first = util.compute_batch_moments(solutions, 1)
  second = util.compute_batch_moments(solutions, 2)

  print('Generating plots and movies')
  results_dir = os.path.join(experiment.DATA_DIRECTORY, name)
  matplotlib.rcParams["axes.formatter.useoffset"] = False
  k = np.arange(solutions.shape[1] - 1) + 1

  fig, _ = plotting.plot_moments(ts, zeroth, first, second)
  fig.savefig(os.path.join(results_dir, 'moments.png'))

  fig, _ = plotting.plot_solution(k, solutions[-1, 1:])
  fig.savefig(os.path.join(results_dir, 'final_solution.png'))

  # Account for 1-indexing.
  plotting.create_solution_animation(ts, k, solutions[:, 1:], os.path.join(results_dir, 'solutions.mp4'))

  print(f'Experiment {name} done. Results saved to {results_dir}')





if __name__ == "__main__":
  main()