# Coding Standards - ERCOT DART Project

> *Based on [PEP 8](https://peps.python.org/pep-0008/), [Black](https://black.readthedocs.io/), and Python community best practices*

## Code Formatting

### **REQUIRED: Use Black Formatter**

All Python code MUST be formatted using [Black](https://black.readthedocs.io/):

```bash
# Format your code before committing
black src/ tests/ scripts/

# Check if code is properly formatted
black --check src/ tests/ scripts/
```

**Configuration**: We use Black's default settings (88 character line length) as defined in `pyproject.toml`.

### **REQUIRED: Use Absolute Imports with `src.` Prefix**

All internal module imports MUST use absolute imports with the `src.` prefix:

```python
# ✅ CORRECT
from src.data.ercot.clients.load import LoadForecastClient
from src.visualization.ercot.ercot_viz import ERCOTBaseViz
from src.etl.ercot.ercot_etl import ERCOTBaseETL

# ❌ INCORRECT - Never use these patterns
from data.ercot.clients.load import LoadForecastClient  # Missing src prefix
from .clients.load import LoadForecastClient            # Relative import
from ..ercot_viz import ERCOTBaseViz                    # Relative import
```

### Import Organization (Following PEP 8)

Group imports in this order with blank lines between groups:

```python
# 1. Standard library imports
from typing import Dict, Optional, List
from datetime import datetime
import os
import sys

# 2. Third-party library imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# 3. Internal module imports (always with src. prefix)
from src.data.ercot.clients.load import LoadForecastClient
from src.models.ercot.exp0.model_adapters import get_model_adapter
```

## Code Quality Standards

### Type Hints (Following PEP 484)

Use type hints for all function signatures and class methods:

```python
# ✅ CORRECT
def process_data(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    return df[df['value'] > threshold]

# ❌ INCORRECT
def process_data(df, threshold=0.5):
    return df[df['value'] > threshold]
```

### Documentation (Following PEP 257)

All public modules, classes, and functions MUST have docstrings:

```python
def get_load_forecast_data(
    self, 
    posted_datetime_from: str, 
    posted_datetime_to: str
) -> pd.DataFrame:
    """Get ERCOT Seven-Day Load Forecast by Model and Weather Zone.
    
    Args:
        posted_datetime_from: Start date in YYYY-MM-DD format
        posted_datetime_to: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame containing the load forecast data
        
    Raises:
        ValueError: If date format is invalid
    """
```

### Naming Conventions (Following PEP 8)

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`  
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes**: `_leading_underscore`
- **Files and modules**: `snake_case.py`

## Why These Standards?

- **Black**: Eliminates code style debates, ensures consistency, and is widely adopted
- **PEP 8**: Python's official style guide, industry standard
- **Absolute imports**: Clarity, reliability, and IDE support
- **Type hints**: Better code documentation, IDE support, and error catching

## Enforcement

### Automated Tools

- **Black**: Code formatting (`black src/ tests/ scripts/`)
- **isort**: Import sorting (`isort src/ tests/ scripts/`)
- **flake8**: Style and quality checking
- **mypy**: Type checking (optional but recommended)

### Pre-commit Hooks

Install and use pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This automatically runs Black, isort, and flake8 on every commit.

### Continuous Integration

All code changes are automatically checked for:
- Black formatting compliance
- Import standards compliance
- PEP 8 compliance via flake8
- Type hint coverage (where applicable)

## Quick Commands

```bash
# Format all code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Check style compliance
flake8 src/ tests/ scripts/

# Check import standards
python scripts/check_imports.py src/**/*.py

# Run all checks
pre-commit run --all-files
```

## For AI Assistants

When working on this codebase:

1. **ALWAYS** format code with Black defaults (88 char line length)
2. **ALWAYS** use absolute imports with `src.` prefix for internal modules
3. **NEVER** use relative imports (`from .` or `from ..`)
4. **ALWAYS** add type hints to function signatures
5. **ALWAYS** include docstrings for public functions and classes
6. Follow PEP 8 naming conventions
7. If you see non-compliant code, fix it immediately

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Black - The Uncompromising Code Formatter](https://black.readthedocs.io/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)

## Project Structure

```
ercot_dart/
├── src/                    # All source code (use src. prefix for imports)
│   ├── data/              # Data collection clients
│   ├── etl/               # Data processing
│   ├── features/          # Feature engineering  
│   ├── models/            # ML models
│   ├── visualization/     # Plotting and analysis
│   └── workflow/          # End-to-end scripts
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

All imports should reference modules relative to the `src/` directory with the `src.` prefix. 