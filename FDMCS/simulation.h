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

inline constexpr int kNumSmallParticles = 1000;

class Simulation {
 public:
  Simulation();
  void AddParticle(int size);
  void DeleteParticle(int idx);
  std::pair<int, int> FindPair(float rate);
  void AddMonomers(int num_monomers);

  std::vector<Particle> GetDistribution();

 private:
  Particle& GetParticle(int idx);
  float CollisionFunction(int first_size, int second_size);

  void InsertParticle(int size, float rate);
  void RemoveParticle(int idx);

  SearchResult FindFirst(float rate);
  SearchResult FindSecond(SearchResult first);

  Particle small_particles[kNumSmallParticles];
  std::vector<Particle> big_particles;
  float total_rate;
  int total_size;
};