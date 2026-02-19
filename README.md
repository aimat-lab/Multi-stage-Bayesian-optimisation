# Multi-Stage Bayesian Optimization for Dynamic Decision-Making in Self-Driving Labs

This repository contains the code for the paper **"Multi-stage Bayesian optimisation for dynamic decision-making in self-driving labs"**.

## Abstract

Self-driving laboratories (SDLs) are combining recent technological advances in robotics, automation, and machine learning based data analysis and decision-making to perform autonomous experimentation toward human-directed goals without requiring any direct human intervention. SDLs are successfully used in materials science, chemistry, and beyond, to optimise processes, materials, and devices in a systematic and data-efficient way.

At present, the most widely used algorithm to identify the most informative next experiment is Bayesian optimisation. While relatively simple to apply to a wide range of optimisation problems, standard Bayesian optimisation relies on a fixed experimental workflow with a clear set of optimisation parameters and one or more measurable objective functions. This excludes the possibility of making on-the-fly decisions about changes in the planned sequence of operations and including intermediate measurements in the decision-making process. Therefore, many real-world experiments need to be adapted and simplified to be converted to the common setting in self-driving labs.

In this paper, we introduce an extension to Bayesian optimisation that allows flexible sampling of multi-stage workflows and makes optimal decisions based on intermediate observables, which we call proxy measurements. We systematically compare the advantage of taking into account proxy measurements over conventional Bayesian optimisation, in which only the final measurement is observed. We find that over a wide range of scenarios, proxy measurements yield a substantial improvement, both in the time to find good solutions and in the overall optimality of found solutions. This not only paves the way to use more complex and thus more realistic experimental workflows in autonomous labs but also to smoothly combine simulations and experiments in the next generation of SDLs.

## Project Structure

```text
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

## Installation

This project is tested with **Python 3.11** (see `environment.yml` for pinned versions).

### Option A (recommended): Conda / mamba

```bash
git clone <REPO_URL>
cd MSBO

# Create and activate the environment
conda env create -f environment.yml
conda activate msbo_env

# Install the package (editable, for development)
pip install -e .
```

To install non-editable (like an end user), use `pip install .` instead.

### Option B: pip-only (virtualenv)

If you prefer not to use Conda, create a virtual environment and install dependencies.
Pinned versions matching `environment.yml` are provided in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Install the pinned dependencies
python -m pip install -r requirements.txt

# Install msbo
python -m pip install -e .
```

## Usage

### Basic Example: 2-Stage Cascade Process

To run a single optimisation loop using the standard two-stage cascade process, execute the following command:

```bash
python -m msbo.main \
  --process standard \
  --complexity 50 2 \
  --start_counter 0
```

This script initialises a synthetic two-stage laboratory environment; draws a random initial dataset; and subsequently executes three comparative optimisation algorithms (MSBO: `msbo_EI_UCB`, standard BO: `global_EI`, and Random Search: `random`) to identify the global optimum.

This specific combination of arguments reproduces the scenario depicted in the top-left panel of Figure 4 in our manuscript.

  - `--process standard`: Selects a fully observable, noiseless two-stage cascade.
  - `--complexity 50 2`: Establishes a scenario wherein the first experimental stage is highly complex and difficult to model (50 seed points), whereas the second stage is relatively simple (2 seed points). This simulates a common real-world bottleneck where upstream proxy preparation is challenging.
  - `--start_counter 0`: Sets the random seed to 0, ensuring reproducible function generation and initial sampling.

#### Reproducing Manuscript Figures

By altering the configuration flags, one can generate the data for the various ablation studies and supplementary figures discussed in the manuscript:

- Figure 4 (Complexity Sweep): Optimise the standard process while varying the combinations of complexities (e.g., 2, 15, 50).

```bash
python -m msbo.main --process standard --complexity 50 2 --start_counter 0
```

- Figure S4 (Three-Stage Cascade): Optimise a deeper sequential process.

```bash
python -m msbo.main --process three_stages --complexity 15 15 5 --start_counter 0
```

- Figure S5 (Partial Observability): Optimise a process wherein intermediate states are only partially visible to the optimiser (i.e., masking is applied).

```bash
python -m msbo.main --process filtered --complexity 50 15 2 --start_counter 0
```

- Figure S6 (Process Noise): Optimise a two-stage cascade with additive process noise to simulate imperfectly reproducible laboratory experiments.

```bash
python -m msbo.main --process noisy_process --complexity 50 2 --start_counter 0
```
Note: The hyperparameters for the optimisation loops (such as iterations and acquisition functions) are defined directly within main.py and are pre-configured to match the experimental conditions reported in our study.

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

1. **msbo_EI_UCB**: Multi-stage optimization with qEI and qUCB acquisition functions
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

## Expected Output
Executing the basic example automatically creates a directory structure at `msbo/results/standard_50_2/0/`. Within this directory, the script generates the following outputs:

- `ground_truth.txt`: Contains the true optimal parameter values and the maximum objective value for the generated landscape.

- JSON Summaries (e.g., msbo_EI_UCB_summary.json): Detailed logs documenting the parameters sampled at each iteration, the selected acquisition functions, and the observed values.

- NumPy Archives (e.g., msbo_EI_UCB_denoised.npz): Extracted arrays of the inputs and denoised objective values, formatted for straightforward post-processing.

- Regret Plots (.png): Automatically generated visualisations depicting the log-regret over the cumulative cost, which facilitate immediate performance comparisons between MSBO, standard BO, and BOFN.

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

