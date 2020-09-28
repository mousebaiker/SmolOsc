#ifndef FDMCS_IO_UTIL

#include <fstream>
#include <string>
#include <filesystem>

#include "simulation.h"

std::string GetFileContents(const std::string& filename)
{
  std::string contents;
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (in)
  {
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
  }
  return(contents);
};

void PrintParticles(const std::vector<Particle>& particles) {
  for (const auto& particle : particles) {
    std::cout << "Particle size: " << particle.size
              << ", count: " << particle.count
              << ", collision rate: " << particle.collision_rate << std::endl;
  }
}


void SaveCheckpoint(Simulation& simulation, std::string output_dir, float simulation_time, int checkpoint_num, std::chrono::nanoseconds elapsed_time) {
  std::filesystem::create_directories(output_dir);

  std::string filename = output_dir + "/" + std::to_string(simulation_time) + ".cpt";
  std::cout << filename << std::endl;
  std::ofstream out(filename, std::ios::out);
  if (out) {
    out << elapsed_time.count() << std::endl;
    for (const auto& particle : simulation.GetDistribution()) {
      out << particle.size << " " << particle.count << " " << particle.collision_rate << std::endl;
    }
    out.close();
  } else {
    std::cerr << "Error code: " << strerror(errno);
  }
}

#define FDMCS_IO_UTIL value
#endif
