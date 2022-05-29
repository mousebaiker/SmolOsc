import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm

def create_solution_animation(data_loader, output_path):
  fig, ax = plt.subplots()
  ln, = ax.loglog([], [])

  def init():
      ax.set_xlim(1, 1000)
      ax.set_ylim(10**(-16), 10**(0))
      return ln,

  def update(frame):
      k, y, ts, _ = data_loader[frame]
      ln.set_data(k, y)
      ax.set_title(f'T={ts:.4}')
      return ln,

  ani = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(data_loader))), interval=30,
                      init_func=init, blit=True)
  ani.save(output_path, writer='ffmpeg')
