#include "simulation.h"

#include <algorithm>
#include <iostream>

Simulation::Simulation() {
  for (int i = 0; i < kNumSmallParticles; i++) {
    small_particles[i] = Particle{0, i, 0};
  }
}

std::vector<Particle> Simulation::GetDistribution() {
  std::vector<Particle> result;
  Particle particle;
  for (int i = 0; i < total_size; i++) {
    particle = GetParticle(i);
    if (particle.count > 0) {
      result.push_back(particle);
    }
  }
  return result;
}

void Simulation::AddParticle(int size) {
  float rate = 0;
  float collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(size, particle.size);
    rate += particle.count * collision_value;
    particle.collision_rate += collision_value;
  }
  InsertParticle(size, rate);
  total_rate += 2 * rate;
}

void Simulation::AddMonomers(int num_monomers) {
  float rate = (num_monomers - 1) * CollisionFunction(1, 1);
  float collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(1, particle.size);
    rate += particle.count * collision_value;
    particle.collision_rate += num_monomers * collision_value;
  }
  InsertParticle(1, rate);
  small_particles[1].count += (num_monomers - 1);
  total_rate += num_monomers * 2 * rate;
}

void Simulation::DeleteParticle(int idx) {
  Particle deleted_particle = GetParticle(idx);
  RemoveParticle(idx);

  float rate = 0;
  float collision_value;
  for (int i = 0; i < total_size; i++) {
    Particle& particle = GetParticle(i);
    collision_value = CollisionFunction(deleted_particle.size, particle.size);
    rate += particle.count * collision_value;
    particle.collision_rate -= collision_value;
  }
  total_rate -= 2 * rate;
}

std::pair<int, int> Simulation::FindPair(float rate) {
  SearchResult first = FindFirst(rate);
  SearchResult second = FindSecond(first);
  return std::pair{first.idx, second.idx};
}

SearchResult Simulation::FindFirst(float rate) {
  int idx;
  Particle particle;
  for (idx = 1; idx < total_size; idx++) {
    particle = GetParticle(idx);
    float group_rate = particle.count * particle.collision_rate;
    if (rate - group_rate <= 0) {
      rate -= int(rate / particle.collision_rate) * particle.collision_rate;
      break;
    }
    rate -= group_rate;
  }

  if (idx >= total_size) {
    std::cout << "Out of bound search" << std::endl;
    return SearchResult{0, rate};
  }

  return SearchResult{idx, rate};
}

SearchResult Simulation::FindSecond(SearchResult first) {
  float rate = first.remaining_rate;
  int first_size = GetParticle(first.idx).size;
  Particle particle;
  float count;
  float group_rate;
  int idx;
  for (idx = 1; idx < total_size; idx++) {
    particle = GetParticle(idx);
    count = particle.count;
    if (idx == first.idx) {
      count -= 1;
    }
    group_rate = count * CollisionFunction(first_size, particle.size);

    if (rate - group_rate <= 0) {
      break;
    }
    rate -= group_rate;
  }

  return SearchResult{idx, rate};
}

Particle& Simulation::GetParticle(int idx) {
  if (idx < kNumSmallParticles) {
    return small_particles[idx];
  }
  return big_particles[idx - kNumSmallParticles];
}

void Simulation::InsertParticle(int size, float rate) {
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

float Simulation::CollisionFunction(int first_size, int second_size) {
  return first_size * second_size;
}
