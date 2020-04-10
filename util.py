import os

import numpy as np

from experiment import DATA_DIRECTORY

def extract_number_from_name(filename):
  num = filename.replace('.npy', '')
  return int(num)

def load_solutions(experiment_name):
  solutions_dir = os.path.join(DATA_DIRECTORY, experiment_name, 'solutions')
  solution_files = sorted(os.listdir(solutions_dir), key=extract_number_from_name)

  iters = []
  solutions = []
  for solution_file in solution_files:
    iters.append(extract_number_from_name(solution_file))
    solutions.append(np.load(os.path.join(solutions_dir, solution_file)))

  return np.array(iters), np.array(solutions)

def compute_batch_moments(solution, order):
  k = np.arange(solution.shape[1])
  return solution @ (k**order)