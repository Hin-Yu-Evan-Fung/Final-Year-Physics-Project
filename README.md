# Final-Year-Physics-Project

## Setup Guide

### For developers

```bash
# 1. Instantiate and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install core dependencies
pip install -ve ".[dev]"

# Add -Ceditable.rebuild=true to automatically rebuild before every run (slow)
# Add -Ccmake.build-type=Release to build in optimised mode
```

### Testing

```bash
pytest -v -s # For Python tests

./build/hsmc_tests # For C++ GTests
```
