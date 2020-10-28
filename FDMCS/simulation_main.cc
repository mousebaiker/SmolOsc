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
    sim->AddMonomers(config.monomer_count());
  }
  return sim;
}



int main(int argc, char const *argv[]) {
  const std::string input = "/gpfs/data/home/a.kalinov/SmolOsc/FDMCS/config/brownian_osc_1000.json";

  SimulationConfiguration config;
  JsonStringToMessage(GetFileContents(input), &config);



  std::unique_ptr<Simulation> simulation = ConstructSimulation(config);
  RunSimulation(*simulation, config.duration(), config.save_options());
  return 0;
}
