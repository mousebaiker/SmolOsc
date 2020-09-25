#include "simulation.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>


Simulation::Simulation() : Simulation(0, std::mt19937{}) {}


Simulation::Simulation(float fragmentation_rate, std::mt19937 rng)
    : rng(rng),
      total_rate(0),
      total_size(0),
      step_counter(0),
      num_particles(0),
      max_num_particles(0),
      fragmentation_rate(fragmentation_rate) {
  for (int i = 0; i < kNumSmallParticles; i++) {
    small_particles[i] = Particle{0, i, 0};
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
  std::uniform_real_distribution<double> frag_dist(0, 1 + fragmentation_rate);
  double rate = pair_dist(rng);
  bool is_aggr = frag_dist(rng) < 1;

  const std::pair<int, int> particles = FindPair(rate);
  long long new_size =
      GetParticle(particles.first).size + GetParticle(particles.second).size;
  if (is_aggr) {
    AddParticle(new_size);
  } else {
    AddMonomers(new_size);
  }
  DeletePair(particles);

  if (step_counter % 1000  == 0) {
    total_rate = CountTotalRate();
  }
  step_counter++;

  if (num_particles <= (max_num_particles / 2)) {
    DuplicateParticles();
  }

  assert(abs(CountTotalRate() - total_rate) < 1);
}


void Simulation::AddParticle(long long size) {
  double rate = 0;
  double collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(size, particle.size);
    rate += collision_value * particle.count;
    particle.collision_rate += collision_value;
  }
  InsertParticle(size, rate);
  total_rate += 2 * rate;
  IncrementParticleCount(1);
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
  IncrementParticleCount(num_monomers);


  // Pairs of newly added monomers were double counted, so we need to fix the
  // total rate.
  double excess = CollisionFunction(1, 1) * num_monomers * (num_monomers - 1);
  total_rate -= excess;
}


void Simulation::DeleteParticle(int idx) {
  Particle deleted_particle = GetParticle(idx);
  RemoveParticle(idx);

  double rate = 0;
  double collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(deleted_particle.size, particle.size);
    rate += collision_value * particle.count;
    particle.collision_rate -= collision_value;
  }
  total_rate -= 2 * rate;
  IncrementParticleCount(-1);
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


void Simulation::DuplicateParticles() {
  int num_sizes = total_size;

  for (int idx = 1; idx < num_sizes; idx++) {
    Particle& particle = GetParticle(idx);
    int count = particle.count;
    int size = particle.size;
    if (size == 1) {
      AddMonomers(count);
    } else {
      for (int j = 0; j < count; j++) {
        AddParticle(size);
      }
    }
  }
  total_rate = CountTotalRate();
}


std::pair<int, int> Simulation::FindPair(double rate) {
  SearchResult first = FindFirst(rate);
  SearchResult second = FindSecond(first);

  return std::pair{first.idx, second.idx};
}


SearchResult Simulation::FindFirst(double rate) {
  double init_rate = rate;
  int idx = 1;
  int last_valid = 1;
  Particle particle;
  for (idx = 1; idx < total_size; idx++) {
    particle = GetParticle(idx);
    if (particle.count > 0) {
      last_valid = idx;
    }

    double group_rate = particle.collision_rate * particle.count;
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
  double rate = first.remaining_rate;
  int first_size = GetParticle(first.idx).size;
  Particle particle;
  double count;
  double group_rate;
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

void Simulation::IncrementParticleCount(int increment) {
  num_particles += increment;
  max_num_particles = std::max(max_num_particles, num_particles);
}


double Simulation::CountTotalRate() {
  double rate = 0;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    rate += particle.collision_rate * particle.count;
  }
  return rate;
}
