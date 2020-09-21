#include <random>
#include <utility>
#include <vector>

typedef struct {
  long long count;
  long long size;
  double collision_rate;
} Particle;

typedef struct {
  int idx;
  double remaining_rate;
} SearchResult;

class Simulation {
 public:
  Simulation();
  Simulation(float fragmentation_rate, std::mt19937 rng);
  void AddParticle(long long size);
  void AddMonomers(long long num_monomers);
  void DeleteParticle(int idx);
  void DeletePair(const std::pair<int, int>& idxs);
  std::pair<int, int> FindPair(double rate);

  void RunSimulationStep();

  std::vector<Particle> GetDistribution();

 private:
  Particle& GetParticle(int idx);
  float CollisionFunction(long long first_size, long long second_size);

  void InsertParticle(long long size, double rate);
  void RemoveParticle(int idx);

  SearchResult FindFirst(double rate);
  SearchResult FindSecond(SearchResult first);

  double CountTotalRate();

  std::vector<Particle> small_particles;
  std::vector<Particle> big_particles;
  double total_rate;
  long long total_size;
  std::mt19937 rng;

  float fragmentation_rate;
};
