#include <random>
#include <utility>
#include <vector>

typedef struct {
  int count;
  int size;
  float collision_rate;
} Particle;

typedef struct {
  int idx;
  float remaining_rate;
} SearchResult;

class Simulation {
 public:
  Simulation();
  Simulation(float fragmentation_rate, std::mt19937 rng);
  void AddParticle(int size);
  void AddMonomers(int num_monomers);
  void DeleteParticle(int idx);
  void DeletePair(const std::pair<int, int>& idxs);
  std::pair<int, int> FindPair(float rate);

  void RunSimulationStep();

  std::vector<Particle> GetDistribution();

  virtual float CollisionFunction(int first_size, int second_size) = 0;

 private:
  Particle& GetParticle(int idx);

  void InsertParticle(int size, float rate);
  void RemoveParticle(int idx);

  SearchResult FindFirst(float rate);
  SearchResult FindSecond(SearchResult first);

  std::vector<Particle> small_particles;
  std::vector<Particle> big_particles;
  double total_rate;
  int total_size;
  std::mt19937 rng;

  float fragmentation_rate;
};

class ConstantKernelSimulation : public Simulation {
  using Simulation::Simulation;
  inline float CollisionFunction(int first_size, int second_size) override {
    return 1.0;
  }
};

class MultiplicationKernelSimulation : public Simulation {
  using Simulation::Simulation;
  inline float CollisionFunction(int first_size, int second_size) override {
    return first_size * second_size / 100000.0;
  }
};
