#ifndef FDMCS_SIMULATION
#define FDMCS_SIMULATION

#include <random>
#include <utility>
#include <vector>
#include <array>
#include <cmath>

typedef struct {
  long long count;
  long long size;
  double collision_rate;
} Particle;

typedef struct {
  int idx;
  double remaining_rate;
} SearchResult;

inline constexpr int kNumSmallParticles = 10000;

class Simulation {
 public:
  Simulation();
  Simulation(float fragmentation_rate, std::mt19937 rng);
  void AddParticle(long long size);
  void AddMonomers(long long num_monomers);
  void DeleteParticle(int idx);
  void DeletePair(const std::pair<int, int>& idxs);
  void DuplicateParticles();

  std::pair<int, int> FindPair(double rate);

  double RunSimulationStep();

  std::vector<Particle> GetDistribution();

  virtual double CollisionFunction(long long first_size, long long second_size) = 0;

 private:
  Particle& GetParticle(int idx);

  void InsertParticle(long long size, double rate);
  void RemoveParticle(int idx);
  inline void IncrementParticleCount(int increment);

  SearchResult FindFirst(double rate);
  SearchResult FindSecond(SearchResult first);

  double CountTotalRate();

  std::array<Particle, kNumSmallParticles> small_particles;
  std::vector<Particle> big_particles;
  double total_rate;
  long long total_size;
  long long num_particles;
  long long num_initial_particles;
  long long max_num_particles;
  std::mt19937 rng;
  double cell_size;

  float fragmentation_rate;

  int step_counter;
};

class ConstantKernelSimulation : public Simulation {
  using Simulation::Simulation;
  inline double CollisionFunction(long long first_size, long long second_size) override {
    return 1.0;
  }
};

class MultiplicationKernelSimulation : public Simulation {
  using Simulation::Simulation;
  inline double CollisionFunction(long long first_size, long long second_size) override {
    return first_size * second_size / 100000.0;
  }
};

class BallisticKernelSimulation : public Simulation {
  using Simulation::Simulation;
  inline double CollisionFunction(long long first_size, long long second_size) override {
    double first = first_size;
    double second = second_size;
    double first_term = pow(pow(first, 1.0/3.0) + pow(second, 1.0/3.0), 2.0);
    double second_term = pow(1.0 / first + 1.0 / second, 0.5);
    return first_term * second_term;
  }
};

class BrownianKernelSimulation : public Simulation {
 public:
  BrownianKernelSimulation(double alpha); 
  BrownianKernelSimulation(float fragmentation_rate, std::mt19937 rng, double alpha);
  inline double CollisionFunction(long long first_size, long long second_size) override {
    double fraction = first_size / (double) second_size;
    double inverse = second_size / (double) first_size;
    double first_term = pow(fraction, alpha_);
    double second_term = pow(inverse, alpha_);
    return first_term + second_term;
  }
 private:
  double alpha_;
};
#endif
