#include "simulation.h"

#include <cmath>

Simulation::Simulation(int num_equations, double dt, float lambda, float alpha):
    concentration(num_equations + 1, 0), update(num_equations + 1, 0), k_n(num_equations + 1, 0),
    k_nn(num_equations + 1, 0), dt(dt), lambda(lambda), alpha(alpha) {
    concentration[1] = 1.0;
}


void Simulation::computeUpdate(const std::vector<double>& concentration, std::vector<double>* update) {
  K_n(concentration, &k_n);
  K_nn(concentration, &k_nn);

  (*update)[1] = -concentration[1] * k_n[1] + lambda / 2 * K_ij_nn(concentration) +
      lambda * concentration[1] * j_K_1j_n(concentration);

  for (int i = 2, n = update->size(); i < n; ++i) {
    (*update)[i] = 0.5 * k_nn[i] - (1 + lambda) * concentration[i] * k_n[i];
  }
}

void Simulation::updateConcentrations(const std::vector<double>& update, std::vector<double>* concentration) {
  for (int i = 1, n = concentration->size(); i < n; ++i) {
    (*concentration)[i] = (*concentration)[i] + dt * update[i];
  }
}


std::vector<double> Simulation::runSimulation(int num_iters) {
  for (int i = 0; i < num_iters; ++i) {
    computeUpdate(concentration, &update);
    updateConcentrations(update, &concentration);
  }
  return concentration;
}


virtual void NaiveSimulation::K_nn(const std::vector<double>& concentration, std::vector<double>* result) final {
  for (int k = 2, n = concentration.size(); k < n; ++k) {
    (*result)[k] = 0;
    for (int i = 1; i < k; ++i) {
      (*result)[k] = concentration[i] * concentration[k-i] *
          (pow(i / float(k - i), alpha) + pow((k - i)/ float(i), alpha));
    }
  }
}

virtual double NaiveSimulation::K_ij_nn(const std::vector<double>& concentration) final {
  double result = 0;
  for (int i = 2, n = concentration.size(); i < n; ++i) {
    for (int j = 2; j < n; ++j) {
      result += (pow(i / float(j), alpha) + pow(j / float(i), alpha)) * (i + j) * concentration[i] * concentration[j];
    }
  }
  return result
}


virtual void K_n(const std::vector<double>& concentration, std::vector<double>* result) final {
  sum_{j >= 1} K_{k, j} * n_{j} -> vector
  """
  result = np.zeros_like(concentration)

  iss = np.arange(len(concentration)).reshape((-1, 1))
  js = np.arange(len(concentration))
  K = (iss / js)**self.alpha + (js / iss)**self.alpha
  K[~np.isfinite(K)] = 0
  return K @ concentration

  

}


virtual double NaiveSimulation::j_K_1j_n(const std::vector<double>& concentration) final {
  double result = 0
  for (int j = 2, n = concentration.size(); j < n; ++j) {
    result += j * (pow(1 / float(j), alpha) + pow(float(j), alpha)) * concentration[j]
  }
  return result
}
