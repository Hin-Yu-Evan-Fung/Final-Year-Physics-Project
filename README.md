# Final-Year-Physics-Project

## Setup Guide

### For developers

#### Setup

```bash
# 1. Compile C++ libraries
cmake -B build         # Run once at the start (-DCMAKE_BUILD_TYPES=Release for optimised mode)
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
