import argparse
import sys
import os

import yaml
import numpy as np
import matplotlib

import experiment
import plotting
import util

def validate_required_param(config, key):
  param = config.get(key)
  if param is None:
    print(f'{key} param is required for experiment execution', file=sys.stderr)
    return False
  return True


def validate_config(config):
  result = True
  result &= validate_required_param(config, 'dt')
  result &= validate_required_param(config, 'lambda')
  result &= validate_required_param(config, 'alpha')
  result &= validate_required_param(config, 'experiment_name')
  result &= validate_required_param(config, 'num_iters')
  result &= validate_required_param(config, 'checkpoint_frequency')

  num_equations = config.get('num_equations')
  path_to_initial = config.get('path_to_initial')
  if num_equations is None and path_to_initial is None:
    print(f'Please specify number of equations (num_equations key) or initial solution (path_to_initial key)', file=sys.stderr)
  result &= num_equations is not None or path_to_initial is not None

  lambda_decay_type = config.get('lambda_decay_type')
  result &= lambda_decay_type is None or validate_required_param(config, 'final_lambda')
  return result


def main():
  parser = argparse.ArgumentParser(description='Run experiment and save the derivative results.')
  parser.add_argument('yaml_config', help='Config YAML file specifying the parameters of experiment')

  args = parser.parse_args()
  with open(args.yaml_config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  if not validate_config(config):
    sys.exit(1)

  initial = None
  path_to_initial = config.get('path_to_initial')
  if path_to_initial is not None:
    initial = np.load(path_to_initial)

  e = experiment.Experiment(dt=config.get('dt'), lmbda=config.get('lambda'),
      alpha=config.get('alpha'), num_equations=config.get('num_equations'),
      initial=initial, final_lambda=config.get('final_lambda'),
      lambda_decay_type=config.get('lambda_decay_type'), use_cuda=config.get('use_cuda'))


  name = config['experiment_name']

  print(f'Starting experiment {name}')
  e.run_experiment(name, config.get('num_iters'),
      config.get('checkpoint_frequency'))

  print('Simulation finished. Loading results')
  iters, solutions = util.load_solutions(name)
  results_dir = os.path.join(experiment.DATA_DIRECTORY, name)

  lambda_history = util.load_param_history(name)
  ts = iters * config['dt']

  print('Computing moments')
  zeroth = util.compute_batch_moments(solutions, 0)
  first = util.compute_batch_moments(solutions, 1)
  second = util.compute_batch_moments(solutions, 2)

  print('Generating plots and movies')
  matplotlib.rcParams["axes.formatter.useoffset"] = False
  k = np.arange(solutions.shape[1] - 1) + 1

  fig, _ = plotting.plot_moments(ts, zeroth, first, second)
  fig.savefig(os.path.join(results_dir, 'moments.png'))

  fig, _ = plotting.plot_solution(k, solutions[-1, 1:])
  fig.savefig(os.path.join(results_dir, 'final_solution.png'))

  fig, _ = plotting.plot_parameter_history(ts, lambda_history, r'$\lambda$')
  fig.savefig(os.path.join(results_dir, 'lambda.png'))

  # Account for 1-indexing.
  plotting.create_solution_animation(ts, k, solutions[:, 1:], os.path.join(results_dir, 'solutions.mp4'))

  print(f'Experiment {name} done. Results saved to {results_dir}')





if __name__ == "__main__":
  main()
