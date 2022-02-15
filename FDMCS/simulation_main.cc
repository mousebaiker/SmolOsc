#include "FDMCS/simulation.pb.h"
#include "FDMCS/simulation.h"
#include "FDMCS/io_util.h"

#include <google/protobuf/util/json_util.h>
#include <iostream>

using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using ::google::protobuf::util::JsonStringToMessage;


nanoseconds RunSimulation(Simulation& simulation, float duration, const SaveOptions& save_options) {
  double simulation_time = 0;
  int last_checkpoint_num = -1;
  int checkpoint_num = 0;

  auto start_time = high_resolution_clock::now();
  while (simulation_time < duration) {
    simulation_time += simulation.RunSimulationStep();

    checkpoint_num = int(simulation_time / save_options.checkpoint_interval());
    if (checkpoint_num > last_checkpoint_num) {
      last_checkpoint_num = checkpoint_num;
      auto elapsed_time = high_resolution_clock::now() - start_time;
      SaveCheckpoint(simulation, save_options.output_dir(), simulation_time, checkpoint_num,
        elapsed_time);
    }
  }
  auto end_time = high_resolution_clock::now();

  return end_time - start_time;
}


std::unique_ptr<Simulation> ConstructSimulation(const SimulationConfiguration& config) {
  std::unique_ptr<Simulation> sim;    
  switch (config.kernel_type()) {
    case SimulationConfiguration::UNKNOWN :
      std::cerr << "Kernel unknown" << std::endl;
      exit(1);
      break;
    case SimulationConfiguration::CONSTANT :
      sim = std::make_unique<ConstantKernelSimulation>(config.fragmentation_rate(), std::mt19937());
      break;
    case SimulationConfiguration::BALLISTIC :
      sim = std::make_unique<BallisticKernelSimulation>(config.fragmentation_rate(), std::mt19937());
      break;
    case SimulationConfiguration::MULTIPLICATION :
      break;
    case SimulationConfiguration::BROWNIAN :
      sim = std::make_unique<BrownianKernelSimulation>(config.fragmentation_rate(), std::mt19937(),
          config.brownian_kernel_params().alpha());
      break;
  }

  if (config.has_load_options()) {
    LoadCheckpoint(*sim, config.load_options().checkpoint_path());
  } else {
    switch (config.initial_conditions().distribution_type()) {
      case InitialConditions::UNKNOWN :
        std::cerr << "Initial conditions are unknown" << std::endl;
        exit(1);
        break;
            
      case InitialConditions::SMALLEST_N :
        long long num_sizes = config.initial_conditions().smallest_n_params().num_sizes();
        long long num_particles_per_size = config.initial_conditions().smallest_n_params().particle_count_for_each_size();
        
        if (num_sizes == 0) {
          std::cerr << "Number of sizes in smallest N cannot be 0." << std::endl;
          exit(1); 
        }
        
        sim->AddMonomers(num_particles_per_size);
        for (long long size = 2; size <= num_sizes; ++size) {
            for (long long num_part = 0; num_part < num_particles_per_size; ++num_part) {
              sim->AddParticle(size);
            }
        }
        break;         
    }
  }
  return sim;
}



int main(int argc, char const *argv[]) {
  if (argc != 2) {
    std::cerr << "Please specify a path to a config and no other arguments." << std::endl;
    exit(2);
  }
  const std::string input = argv[1];

  SimulationConfiguration config;
  JsonStringToMessage(GetFileContents(input), &config);



  std::unique_ptr<Simulation> simulation = ConstructSimulation(config);
  RunSimulation(*simulation, config.duration(), config.save_options());
  return 0;
}
