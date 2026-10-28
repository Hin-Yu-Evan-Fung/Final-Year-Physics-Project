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
