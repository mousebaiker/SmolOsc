# SmolOsc: A efficient solver of Smoluchowski equations for aggregation-fragmenation kinetics

This package contains two primary methods to solve Smoluchowski equations:

1. Finite-difference method with low-rank kernel acceleration based on [[1](#references)].
1. Monte Carlo method based on Fast Direct Monte Carlo Simulation (FDMCS) [[2](#references)].

### Contributions
1. We augment FDMCS method to be suitable for the fragmentation kinetics by choosing an event type before picking the pair of particles.
2. We introduce particle grouping in FDMCS to speed up small-size particle updates, including a heavy batch monomer operation after particle fragmentation event.


## Finite-difference methods

### Requirements

The method is implemented in Python. Any Python >= 3.7 should be fine. We have used Python 3.8 in our experiments.

TODO: List of requirements.txt


### Running an experiment

Every experiment is defined by a its own YAML config. `config` directory contains some examples of previously used configs. Specifically, `config/constant_32767` and `config/ballistic_5000` are good starting points on how to define your own experiment.

Once the config is created, running an experiment is as easy as

```bash
python run_experiment.py <PATH_TO_CONFIG>
```

For example, using earlier `config/constant_32767`

```bash
python run_experiment.py config/constant_32767
```

This will create a `experiment.DATA_DIRECTORY` directory *(Yeah, sorry, global variable)* and place the results into its own directory under the name defined in the config.

The produced results include:

* `solutions` directory with solutions files timestamped by iteration number.
* `final_solution.png` -- a figure of particle distribution in the final solution.
* `lambda.png`, `lambda.npy` -- a figure depicting an evolution of lambda (fragmentation rate) parameter through time and corresponding numpy array of lambda values.
* `moments.png`, `moments_cutoff.png` -- figures with evolution of moments through time. `cutoff` version removes skips several first steps when the moment variance might be very big.

* `solutions.mp4` -- video showing how size distribuitons change with time.

### Adding a new kernel

### Running tests






# References

[1]: Matveev, Sergey A., Alexander P. Smirnov, and E. E. Tyrtyshnikov. "A fast numerical method for the Cauchy problem for the Smoluchowski equation." Journal of Computational Physics 282 (2015): 23-32.

[2]: Kruis, F.Einar, et al. “Direct Simulation Monte Carlo Method for Particle Coagulation and Aggregation.” Aiche Journal, vol. 46, no. 9, 2000, pp. 1735–1742.
