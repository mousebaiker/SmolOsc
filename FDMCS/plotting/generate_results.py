import os

from data_loader import DataLoader
from plotting import create_solution_animation

input_dir_prefix = "C:/Users/Aleksei/Documents/Skoltech/kinetics/project/FDMCS/output/"

def get_times(data_loader, breakpoints):
  result = []
  for i in range(len(data_loader)):
      sizes, counts, time, duration = data_loader[i]
      for breakpoint in breakpoints:
          if abs(time - breakpoint) < 1e-5:
              result.append((time, duration, sizes, counts))
  sizes, counts, time, duration = data_loader[-1]
  result.append((time, duration, sizes, counts))
  return result

def print_statistics(data_points):
  ts, dur, _, _ = zip(*data_points)
  ts = [f'{t:#.7g}' for t in ts]
  dur = [f'{d / 10**9:#.7g}' for d in dur]

  print('Simulation time:', *ts, sep='\t')
  print('CPU time:\t', *dur, sep='\t')


def main():

  simul_sizes = [10**3, 10**4, 10**5]

  print('Computing statistics...\n')
  for simul_size in simul_sizes:
    input_dir = os.path.join(input_dir_prefix, f'constant_{simul_size}')
    loader = DataLoader(input_dir)
    breakpoints = [1, 10]
    points = get_times(loader, breakpoints)

    print('Simulation size:', simul_size)
    print_statistics(points)

  print('Generating animations')
  for simul_size in simul_sizes:
    input_dir = os.path.join(input_dir_prefix, f'constant_{simul_size}')
    loader = DataLoader(input_dir)

    output_path = input_dir + '.mp4'
    create_solution_animation(loader, output_path)
    print('Output generated at:', simul_size)


if __name__ == '__main__':
  main()
