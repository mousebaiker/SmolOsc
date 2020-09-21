#include "simulation.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

inline constexpr int kNumSmallParticles = 1000;

Simulation::Simulation() : Simulation(0, std::mt19937{}) {}

Simulation::Simulation(float fragmentation_rate, std::mt19937 rng)
    : rng(rng),
      total_rate(0),
      total_size(0),
      step_counter(0),
      fragmentation_rate(fragmentation_rate) {
  small_particles.reserve(kNumSmallParticles);
  for (int i = 0; i < kNumSmallParticles; i++) {
    small_particles.push_back({0, i, 0});
  }
}

std::vector<Particle> Simulation::GetDistribution() {
  std::vector<Particle> result;
  Particle particle;
  for (int i = 0; i < total_size; i++) {
    particle = GetParticle(i);
    if (particle.count != 0) {
      result.push_back(particle);
    }
  }
  return result;
}

void Simulation::RunSimulationStep() {
  std::uniform_real_distribution<double> pair_dist(0, total_rate);
  double rate = pair_dist(rng);

  const std::pair<int, int> particles = FindPair(rate);
  long long new_size =
      GetParticle(particles.first).size + GetParticle(particles.second).size;

  std::uniform_real_distribution<double> frag_dist(0, 1 + fragmentation_rate);
  bool is_aggr = frag_dist(rng) < 1;
  if (is_aggr) {
    AddParticle(new_size);
  } else {
    AddMonomers(new_size);
  }
  DeletePair(particles);

  assert(abs(CountTotalRate() - total_rate) < 1);

  if (step_counter % 1000  == 0) {
    total_rate = CountTotalRate();
  }
  step_counter++;
}


double Simulation::CountTotalRate() {
  double rate = 0;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    rate += particle.collision_rate * particle.count;
  }
  return rate;
}


void Simulation::AddParticle(long long size) {
  double rate = 0;
  float collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(size, particle.size);
    rate += collision_value * particle.count;
    particle.collision_rate += collision_value;
  }
  InsertParticle(size, rate);
  total_rate += 2 * rate;
}

void Simulation::AddMonomers(long long num_monomers) {
  double rate = CollisionFunction(1, 1) * (num_monomers - 1);
  double collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(1, particle.size);
    rate += collision_value * particle.count;
    particle.collision_rate += collision_value * num_monomers;
  }
  InsertParticle(1, rate);
  small_particles[1].count += (num_monomers - 1);
  total_rate += rate * num_monomers * 2;

  // Pairs of newly added monomers were double counted, so we need to fix the
  // total rate.
  double excess = CollisionFunction(1, 1) * num_monomers * (num_monomers - 1);
  total_rate -= excess;
}

void Simulation::DeleteParticle(int idx) {
  Particle deleted_particle = GetParticle(idx);
  RemoveParticle(idx);

  float rate = 0;
  float collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(deleted_particle.size, particle.size);
    rate += collision_value * particle.count;
    particle.collision_rate -= collision_value;
  }
  total_rate -= 2 * rate;
}

void Simulation::DeletePair(const std::pair<int, int>& idxs) {
  int first = idxs.first;
  int second = idxs.second;
  if (first < second) {
    std::swap(first, second);
  }
  DeleteParticle(first);
  DeleteParticle(second);
}

std::pair<int, int> Simulation::FindPair(double rate) {
  SearchResult first = FindFirst(rate);
  SearchResult second = FindSecond(first);

  return std::pair{first.idx, second.idx};
}

SearchResult Simulation::FindFirst(double rate) {
  int idx = 1;
  int last_valid = 1;
  Particle particle;
  for (idx = 1; idx < total_size; idx++) {
    particle = GetParticle(idx);
    if (particle.count > 0) {
      last_valid = idx;
    }

    float group_rate = particle.collision_rate * particle.count;
    if (rate - group_rate <= 0) {
      rate -= particle.collision_rate * int(rate / particle.collision_rate);
      break;
    }
    rate -= group_rate;
  }

  if (idx >= total_size) {
    std::cerr << "Out of bound search" << std::endl;
    return SearchResult{0, rate};
  }

  return SearchResult{last_valid, rate};
}

SearchResult Simulation::FindSecond(SearchResult first) {
  float rate = first.remaining_rate;
  int first_size = GetParticle(first.idx).size;
  Particle particle;
  float count;
  float group_rate;
  int idx = 1;
  int last_valid = 1;
  for (idx = 1; idx < total_size; idx++) {
    particle = GetParticle(idx);
    count = particle.count;
    if (idx == first.idx) {
      count -= 1;
    }
    if (count > 0) {
      last_valid = idx;
    }


    group_rate = CollisionFunction(first_size, particle.size) * count;
    if (rate - group_rate <= 0) {
      break;
    }
    rate -= group_rate;
  }

  return SearchResult{last_valid, rate};
}

Particle& Simulation::GetParticle(int idx) {
  if (idx < kNumSmallParticles) {
    return small_particles[idx];
  }
  return big_particles[idx - kNumSmallParticles];
}

void Simulation::InsertParticle(long long size, double rate) {
  if (size < kNumSmallParticles) {
    small_particles[size].count += 1;
    small_particles[size].collision_rate = rate;
    total_size = std::max(total_size, size + 1);
  } else {
    Particle particle{1, size, rate};
    big_particles.push_back(particle);
    total_size = kNumSmallParticles + big_particles.size();
  }
}

void Simulation::RemoveParticle(int idx) {
  if (idx < kNumSmallParticles) {
    small_particles[idx].count -= 1;
  } else {
    idx -= kNumSmallParticles;
    std::swap(big_particles[idx], big_particles.back());
    big_particles.pop_back();
    total_size = kNumSmallParticles + big_particles.size();
  }
}

double Simulation::CollisionFunction(long long first_size,
                                    long long second_size) {
  return first_size * second_size / 10000000.0;
}
