#include "simulation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

#include <iostream>

inline constexpr int kNumSmallParticles = 1000;

inline bool operator==(const Particle& lhs, const Particle& rhs) {
  return lhs.size == rhs.size && lhs.count == rhs.count &&
         abs(lhs.collision_rate - rhs.collision_rate) < 1e-8;
}

void PrintParticles(const std::vector<Particle>& particles) {
  for (const auto& particle : particles) {
    std::cout << "Particle size: " << particle.size
              << ", count: " << particle.count
              << ", collision rate: " << particle.collision_rate << std::endl;
  }
}

TEST(SimulationTest, AddParticleWorks) {
  Simulation simulation;
  simulation.AddParticle(1);

  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/1, /*size=*/1, /*rate=*/0}));

  simulation.AddParticle(1);
  particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/2, /*size=*/1, /*rate=*/1}));

  simulation.AddParticle(2);
  particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/2, /*size=*/1, /*rate=*/3},
                             Particle{/*count=*/1, /*size=*/2, /*rate=*/4}));

  simulation.AddParticle(10000);
  particles = simulation.GetDistribution();
  EXPECT_THAT(particles,
              UnorderedElementsAre(
                  Particle{/*count=*/2, /*size=*/1, /*rate=*/10003},
                  Particle{/*count=*/1, /*size=*/2, /*rate=*/20004},
                  Particle{/*count=*/1, /*size=*/10000, /*rate=*/40000}));
}

TEST(SimulationTest, FindPairWorks) {
  Simulation simulation;
  simulation.AddParticle(1);
  simulation.AddParticle(1);
  simulation.AddParticle(2);
  simulation.AddParticle(10000);

  std::vector<std::pair<int, int>> pairs{
      simulation.FindPair(0.0),     simulation.FindPair(2.0),
      simulation.FindPair(5000.0),  simulation.FindPair(10003.5),
      simulation.FindPair(10005.0), simulation.FindPair(15000.0),
      simulation.FindPair(20007.0), simulation.FindPair(20009.0),
      simulation.FindPair(25000.0), simulation.FindPair(45000.0),
      simulation.FindPair(55000.0), simulation.FindPair(65000.0),
  };

  EXPECT_THAT(
      pairs,
      ElementsAre(
          std::pair{1, 1}, std::pair{1, 2}, std::pair{1, kNumSmallParticles},
          std::pair{1, 1}, std::pair{1, 2}, std::pair{1, kNumSmallParticles},
          std::pair{2, 1}, std::pair{2, 1}, std::pair{2, kNumSmallParticles},
          std::pair{kNumSmallParticles, 1}, std::pair{kNumSmallParticles, 1},
          std::pair{kNumSmallParticles, 2}));
}

TEST(SimualtionTest, AddMonomersWorksWithSingles) {
  Simulation simulation;
  simulation.AddParticle(2);
  simulation.AddParticle(10000);

  simulation.AddMonomers(2);
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles,
              UnorderedElementsAre(
                  Particle{/*count=*/2, /*size=*/1, /*rate=*/10003},
                  Particle{/*count=*/1, /*size=*/2, /*rate=*/20004},
                  Particle{/*count=*/1, /*size=*/10000, /*rate=*/40000}));
}

TEST(SimulationTest, AddMonomersWorksOnMultiples) {
  Simulation simulation;
  simulation.AddParticle(2);
  simulation.AddParticle(2);
  simulation.AddParticle(1);

  simulation.AddMonomers(2);
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/3, /*size=*/1, /*rate=*/6},
                             Particle{/*count=*/2, /*size=*/2, /*rate=*/10}));
}

TEST(SimulationTest, DeleteParticleWorks) {
  Simulation simulation;
  simulation.AddParticle(1);
  simulation.AddParticle(1);
  simulation.AddParticle(2);
  simulation.AddParticle(10000);

  simulation.DeleteParticle(2);
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles,
              UnorderedElementsAre(
                  Particle{/*count=*/2, /*size=*/1, /*rate=*/10001},
                  Particle{/*count=*/1, /*size=*/10000, /*rate=*/20000}));

  simulation.DeleteParticle(1);
  particles = simulation.GetDistribution();
  EXPECT_THAT(particles,
              UnorderedElementsAre(
                  Particle{/*count=*/1, /*size=*/1, /*rate=*/10000},
                  Particle{/*count=*/1, /*size=*/10000, /*rate=*/10000}));

  simulation.DeleteParticle(1000);
  particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/1, /*size=*/1, /*rate=*/0}));

  simulation.DeleteParticle(1);
  particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre());
}