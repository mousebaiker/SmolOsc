#include "simulation.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::duration_cast;

namespace {

void PrintParticles(const std::vector<Particle>& particles) {
  for (const auto& particle : particles) {
    std::cout << "Particle size: " << particle.size
              << ", count: " << particle.count
              << ", collision rate: " << particle.collision_rate << std::endl;
  }
}

nanoseconds recordDuration(Simulation& simulation, int num_iterations) {
  auto start = high_resolution_clock::now();

  for (int i = 0; i < num_iterations; i++) {
    simulation.RunSimulationStep();
  }

  auto end = high_resolution_clock::now();
  return end - start;
}
} // namespace

int main(int argc, char const *argv[]) {

  std::vector<int> initial_monomers{1000, 10000, 100'000, 1'000'000};

  const int num_iterations = 1 * 1000 * 1;

  std::cout << std::setprecision(10);

  std::cout << "Completing " << num_iterations << " for each experiment." << std::endl;
  for (auto num_monomers : initial_monomers) {
    const float fragmentation_rate = 0.2;
    BrownianKernelSimulation simulation(fragmentation_rate, std::mt19937(), 0.9);
    simulation.AddMonomers(num_monomers);
    nanoseconds running_time = recordDuration(simulation, num_iterations);
    std::cout << "  Running time for " << num_monomers << " monomers is " <<
        duration_cast<microseconds>(running_time).count()  << " microseconds."<< std::endl;
  }
  return 0;
}
