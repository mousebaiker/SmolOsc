#include "simulation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::AnyOf;

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

int CountParticles(const std::vector<Particle>& particles) {
  int count = 0;
  for(const auto& particle : particles) {
    count += particle.size * particle.count;
  }
  return count;
}


class TestSimulation : public Simulation {
  using Simulation::Simulation;
  inline double CollisionFunction(long long first_size, long long second_size) override {
    return first_size * second_size;
  }
};


TEST(SimulationTest, AddParticleWorks) {
  TestSimulation simulation;
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
  TestSimulation simulation;
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
  TestSimulation simulation;
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
  TestSimulation simulation;
  simulation.AddParticle(2);
  simulation.AddParticle(2);
  simulation.AddParticle(1);

  simulation.AddMonomers(2);
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/3, /*size=*/1, /*rate=*/6},
                             Particle{/*count=*/2, /*size=*/2,
                                      /*rate=*/10}));
}

TEST(SimulationTest, DeleteParticleWorks) {
  TestSimulation simulation;
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

TEST(SimulationTest, DeletePairDeletesSmall) {
  TestSimulation simulation;
  simulation.AddParticle(1);
  simulation.AddParticle(2);

  simulation.DeletePair(std::pair{1, 2});
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, IsEmpty());
}

TEST(SimulationTest, DeletePairDeletesSameSmall) {
  TestSimulation simulation;
  simulation.AddParticle(1);
  simulation.AddParticle(1);

  simulation.DeletePair(std::pair{1, 1});
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, IsEmpty());
}

TEST(SimulationTest, DeletePairDeletesBig) {
  TestSimulation simulation;
  simulation.AddParticle(10000);
  simulation.AddParticle(10001);

  simulation.DeletePair(std::pair{kNumSmallParticles, kNumSmallParticles + 1});
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, IsEmpty());
}

TEST(SimulationTest, RunSimulationStepNoFragmentation) {
  TestSimulation simulation;
  simulation.AddMonomers(6);

  simulation.RunSimulationStep();
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/4, /*size=*/1, /*rate=*/5},
                             Particle{/*count=*/1, /*size=*/2, /*rate=*/8}));

  simulation.RunSimulationStep();
  particles = simulation.GetDistribution();
  PrintParticles(particles);
  EXPECT_THAT(particles, AnyOf(
    UnorderedElementsAre(
      Particle{/*count=*/2, /*size=*/1, /*rate=*/5},
      Particle{/*count=*/2, /*size=*/2, /*rate=*/8}
    ),
    UnorderedElementsAre(
      Particle{/*count=*/3, /*size=*/1, /*rate=*/5},
      Particle{/*count=*/1, /*size=*/3, /*rate=*/9}
    )));
}

TEST(SimulationTest, RunSimulationStepFragmentation) {
  TestSimulation simulation(/*fragmentation_rate=*/10000.0, std::mt19937(0));
  simulation.AddParticle(10000);
  simulation.AddParticle(20000);

  simulation.RunSimulationStep();
  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(Particle{
                             /*count=*/30000, /*size=*/1, /*rate=*/29999}));

  simulation.RunSimulationStep();
  particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(Particle{
                             /*count=*/30000, /*size=*/1, /*rate=*/29999}));
}

TEST(SimulationTest, RunSimulationWithDuplication) {
  TestSimulation simulation;
  simulation.AddMonomers(4);

  simulation.RunSimulationStep();
  simulation.RunSimulationStep();
  std::vector<Particle>  particles = simulation.GetDistribution();

  EXPECT_EQ(CountParticles(particles), 8);
}


TEST(SimulationTest, DuplicateParticles) {
  TestSimulation simulation;
  simulation.AddParticle(1);
  simulation.AddParticle(1);
  simulation.AddParticle(2);
  simulation.AddParticle(10000);


  simulation.DuplicateParticles();

  std::vector<Particle> particles = simulation.GetDistribution();
  EXPECT_THAT(particles, UnorderedElementsAre(
                             Particle{/*count=*/4, /*size=*/1, /*rate=*/20007},
                             Particle{/*count=*/2, /*size=*/2, /*rate=*/40012},
                             Particle{/*count=*/1, /*size=*/10000, /*rate=*/100 * 1000 * 1000 + 80 * 1000},
                             Particle{/*count=*/1, /*size=*/10000, /*rate=*/100 * 1000 * 1000 + 80 * 1000}));
}
