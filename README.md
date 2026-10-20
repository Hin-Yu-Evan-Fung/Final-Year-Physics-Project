# Final-Year-Physics-Project

## Setup Guide

### For developers

```bash
# 1. Instantiate and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install core dependencies
pip install --no-build-isolation -ve "./hsmc[dev]"

# Add -Ceditable.rebuild=true to automatically rebuild before every run (slow)
```

### Testing

```bash
pytest -v -s # For Python tests

./build/hsmc_tests # For C++ GTests
```
