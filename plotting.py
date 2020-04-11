import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

def plot_moments(ts, zeroth, first, second):
  fig, axes = plt.subplots(3, figsize=(15, 15), sharex=True)

  axes[0].plot(ts, zeroth)
  axes[0].set_title('Total concentration',fontsize=16)
  axes[1].plot(ts, first)
  axes[1].set_title('First moment of concentration', fontsize=16)
  axes[2].plot(ts, second)
  axes[2].set_title('Second moment of concentration', fontsize=16)

  axes[2].set_xlabel('Time, s', fontsize=14)
  axes[0].tick_params(axis='both', which='major', labelsize=14)
  axes[0].tick_params(axis='both', which='minor', labelsize=14)
  axes[1].tick_params(axis='both', which='major', labelsize=14)
  axes[1].tick_params(axis='both', which='minor', labelsize=14)
  axes[2].tick_params(axis='both', which='major', labelsize=14)
  axes[2].tick_params(axis='both', which='minor', labelsize=14)
  return fig, axes


def plot_solution(k, solution, analytical=None):
  fig, ax = plt.subplots(1, figsize=(15, 10))
  ax.loglog(k, solution, label='Numerical solution')
  if analytical is not None:
    ax.loglog(k, analytical, label='Analytical')
  ax.set_xlabel('Cluster size', fontsize=14)
  ax.set_ylabel('Concentration', fontsize=14)
  ax.legend(fontsize=14)
  ax.grid()
  ax.tick_params(axis='both', which='major', labelsize=14)
  ax.tick_params(axis='both', which='minor', labelsize=14)
  return fig, ax


def plot_parameter_history(ts, history, parameter_name):
  fig, ax = plt.subplots(1, figsize=(15, 10))
  ax.plot(ts, history, label=parameter_name, lw=3)
  ax.set_xlabel('Time, s', fontsize=14)
  ax.legend(fontsize=14)
  ax.grid()
  ax.tick_params(axis='both', which='major', labelsize=14)
  ax.tick_params(axis='both', which='minor', labelsize=14)
  return fig, ax


def create_solution_animation(ts, k, solutions, name, analytical=None, loglog=True):
  fig, ax = plt.subplots()
  if loglog:
    ln, = ax.loglog([], [])
  else:
    ln, = ax.semilogy([], [])
  if analytical is not None:
    ax.loglog(k, analytical, label='Analytical')

  def init():
      ax.set_xlim(1, k[-1])
      ax.set_ylim(10**(-16), 10**(0))
      return ln,

  def update(frame):
      y = solutions[frame]
      ln.set_data(k, y)
      ax.set_title('T=' + str(ts[frame]))
      return ln,

  ani = FuncAnimation(fig, update, frames=np.arange(len(solutions)), interval=30,
                      init_func=init, blit=True)
  ani.save(name, writer='ffmpeg')
