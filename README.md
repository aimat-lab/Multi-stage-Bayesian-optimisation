# Multi-Stage Bayesian Optimization for Dynamic Decision-Making in Self-Driving Labs

This repository contains the code for the paper **"Multi-stage Bayesian optimisation for dynamic decision-making in self-driving labs"**.

## Abstract

Self-driving laboratories (SDLs) are combining recent technological advances in robotics, automation, and machine learning based data analysis and decision-making to perform autonomous experimentation toward human-directed goals without requiring any direct human intervention. SDLs are successfully used in materials science, chemistry, and beyond, to optimise processes, materials, and devices in a systematic and data-efficient way.

At present, the most widely used algorithm to identify the most informative next experiment is Bayesian optimisation. While relatively simple to apply to a wide range of optimisation problems, standard Bayesian optimisation relies on a fixed experimental workflow with a clear set of optimisation parameters and one or more measurable objective functions. This excludes the possibility of making on-the-fly decisions about changes in the planned sequence of operations and including intermediate measurements in the decision-making process. Therefore, many real-world experiments need to be adapted and simplified to be converted to the common setting in self-driving labs.

In this paper, we introduce an extension to Bayesian optimisation that allows flexible sampling of multi-stage workflows and makes optimal decisions based on intermediate observables, which we call proxy measurements. We systematically compare the advantage of taking into account proxy measurements over conventional Bayesian optimisation, in which only the final measurement is observed. We find that over a wide range of scenarios, proxy measurements yield a substantial improvement, both in the time to find good solutions and in the overall optimality of found solutions. This not only paves the way to use more complex and thus more realistic experimental workflows in autonomous labs but also to smoothly combine simulations and experiments in the next generation of SDLs.

## Project Structure

```
msbo/
├── main.py                    # Main entry point for running experiments
├── config/                    # Configuration files for different process types
│   ├── standard.yml
│   ├── noisy_process.yml
│   ├── noisy.yml
│   └── filtered.yml
├── optimizer/                 # Bayesian optimization implementation
│   ├── BO.py                 # Main BayesOptimizer class
│   ├── models.py             # Cascade GP model
│   ├── acquisition_function.py # Acquisition function definitions
│   ├── lab_interface.py       # Interface to laboratory/oracle
│   └── utils.py              # Utility functions
├── simulator/                 # Multi-stage process simulation
│   ├── laboratory.py         # Laboratory/Oracle implementation
│   ├── process.py            # Multi-step process definition
│   ├── function_generator.py # Neural network function generation
│   ├── grad_optim.py         # Gradient-based optimization
│   ├── inventory.py          # Sample inventory management
│   └── utils.py              # Utility functions
└── results/                   # Output directory for experiment results
```

## Usage

### Basic Example

```bash
python main.py \
  --process standard \
  --complexity 50 2 \
  --start_counter 0
```

### Command Line Arguments

- `-p, --process`: Name of configuration file of the process to optimize
- `-c, --complexity`: Complexity parameters for each stage
- `-sc, --start_counter`: Starting seed/counter (default: 0)

### Configuration Files

Configuration files in the `config/` directory define:
- Input and output dimensions for each stage
- Measurement modes (identity, filter, etc.)
- Process noise levels (if applicable)
- Other process-specific parameters

## Running Experiments

The main script runs multiple optimization algorithms and compares their performance:

1. **msopt_EI_UCB**: Multi-stage optimization with qEI and qUCB acquisition functions
2. **random**: Random search baseline
3. **global_EI**: Global optimization with qEI

Results are saved to `results/{process}_{complexity}/` directory with:
- Ground truth optimal values
- Optimization summary statistics
- Denoised performance metrics
- Performance plots (regret curves)

## Requirements

- PyTorch
- BoTorch
- GPyTorch
- NumPy
- PyYAML
- Pandas
- Matplotlib

## Output

For each experiment seed, the following files are generated in the results directory:

- `ground_truth.txt`: Optimal parameter values and objective value
- `{optimizer_name}_summary.json`: Detailed optimization results
- `{optimizer_name}_denoised.npz`: Denoised objective values
- `{optimizer_name}_*.png`: Performance and regret plots

## Authors

- **Luca Torresi** - *Main implementation and research*
- **Pascal Friederich** - *Supervision and research*

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{torresi2025multistagebayesianoptimisationdynamic,
      title={Multi-stage Bayesian optimisation for dynamic decision-making in self-driving labs}, 
      author={Luca Torresi and Pascal Friederich},
      year={2025},
      eprint={2512.15483},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.15483}, 
}
```

