import numpy as np

def analytical_solution(k, lmbda):
    return lmbda /(np.sqrt(np.pi) * k**(3/2)) * np.exp(-lmbda**2 * k)


def compute_simulation_history():
  history = []
  num_iterations = 15000 * 100
  num_iterations_per_step = 1000
  num_steps = num_iterations // num_iterations_per_step
  k = np.arange(0, 15001)
  ts = np.arange(num_steps) * (osc_big_sim.dt * num_iterations_per_step)
  start = time.time()
  for i in range(num_steps):
      conc = osc_big_sim.run_simulation(num_iterations_per_step)
      history.append(conc)
  end = time.time()


  osc_big_cons.append(np.sum(conc))
  ksq = k**2
  osc_big_firsts.append(k @ conc)
  osc_big_seconds.append(ksq @ conc)
