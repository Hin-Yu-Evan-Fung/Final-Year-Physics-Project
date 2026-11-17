# Final-Year-Physics-Project

## Setup Guide

### For developers

#### Setup

```bash
# Make sure you have brew installed
brew install cmake, llvm

# 1. Compile C++ libraries
# For mac users:
cmake --preset mac-llvm
# For linux users:
cmake -B build         # (-DCMAKE_BUILD_TYPES=Release for optimised mode)
# Then for all users:
cmake --build build -j # Run every time there is an update to the C++ code

# 2. Instantiate and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install required libraries
pip install -r requirements.txt
pip install -e .
```

#### Updating Python libraries

```bash
pip freeze --exclude-editable > requirements.txt
```

### Testing

```bash
pytest -s          # For Python tests
./build/hsmc_tests # For C++ GTests
```

### Benchmarking

```bash
python scripts/run_bench.py -n 216 -l 10.0 -s 10000
```

### Storing Jupyter Notebooks

#### Storing at markdown files

```bash
jupytext --to markdown [filepath]
```

#### Converting back to local .ipynb files

```bash
jupytext --to notebook [filepath]
```

### Simulations

#### 1. Running a single simulation

Run a single trajectory by specifying a configuration file, output directory, and RNG seed:

```bash
python run.py --config configs/fcc_hcp.json --outdir [outdir] --seed [seed]
```

Outputs and a copy of the resolved configuration are saved to `[outdir]/seed_[seed]/`.

#### 2. Running parallel simulation

##### Using GNU Parallel

Make sure you have GNU parallel installed:

```bash
mkdir -p [outdir]
seq 1 9 | parallel -j 8 "python -u scripts/run_sim.py --config [config] --outdir [outdir] --seed {} > [outdir]/seed_{}.log 2>&1"
```

* **`-j 8`**: Limits execution to 8 concurrent processes (one per CPU core).
* **`python -u`**: Flushes console output immediately so log files update in real time.
* **`> .../seed_{}.log 2>&1`**: Captures both standard output and any error tracebacks per replica.

##### Using Tmux

```bash
tmux new-session -d -s lsmc -n seed_1 "python scripts/run_sim.py --config [config] --outdir [outdir] --seed 1; read"
for s in {2..8}; do
    tmux new-window -t lsmc -n "seed_$s" "python scripts/run_sim.py --config [config] --outdir [outdir] --seed $s; read"
done
tmux attach -t lsmc
```
