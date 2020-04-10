#include <vector>

class Simulation {
 public:
  Simulation(int num_equations, double dt, float lambda, float alpha);

  virtual void K_nn(const std::vector<double>& concentration, std::vector<double>* result);
  virtual double K_ij_nn(const std::vector<double>& concentration);
  virtual void K_n(const std::vector<double>& concentration, std::vector<double>* result);
  virtual double j_K_1j_n(const std::vector<double>& concentration);

  void computeUpdate(const std::vector<double>& concentration, std::vector<double>* update);
  void updateConcentrations(const std::vector<double>& update, std::vector<double>* concentration);
  std::vector<double> runSimulation(int num_iters);
 private:
  std::vector<double> concentration;
  std::vector<double> update;
  std::vector<double> k_n;
  std::vector<double> k_nn;
  double dt;
  double lambda;
  double alpha;
};

class NaiveSimulation: public Simulation {
 public:
   virtual void K_nn(const std::vector<double>& concentration, std::vector<double>* result) final;
   virtual double K_ij_nn(const std::vector<double>& concentration) final;
   virtual void K_n(const std::vector<double>& concentration, std::vector<double>* result) final;
   virtual double j_K_1j_n(const std::vector<double>& concentration) final;
};
