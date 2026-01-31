# Getting Started

Welcome to `ts-stat-tests`, a comprehensive Python library designed for rigorous time-series statistical testing. This guide provides an overview of the library's core architecture and demonstrates how to perform your first statistical tests.


## 1. Installation

The recommended way to install [`ts-stat-tests`][ts-stat-tests] is via [`uv`][uv], a high-performance Python package installer.


### Using `uv` (recommended)

To add `ts-stat-tests` to your project:

```sh {.sh .bash linenums="1" title="Terminal"}
uv add ts-stat-tests
```


### Using `pip`

Alternatively, you can install it using standard `pip`:

```sh {.sh .bash linenums="1" title="Terminal"}
pip install ts-stat-tests
```


## 2. Core Concepts

The library is structured into functional modules (e.g., `stationarity`, `normality`, `correlation`), each offering three levels of interaction:

1.  **Dispatchers**: Versatile entry points (e.g., `stationarity()`) that wrap multiple specific algorithms under a single interface.
2.  **Boolean Checks**: Convenience functions (e.g., `is_stationary()`) that return a dictionary with a Boolean result based on a significance level (`alpha`).
3.  **Specific Algorithms**: Direct access to underlying implementations (e.g., `adf()`, `kpss()`) for advanced configuration.


## 3. Quickstart Example

Let's perform a stationarity test on some sample airline data.

```py {.py .python linenums="1" title="Stationarity Example"}
import numpy as np
from ts_stat_tests.stationarity import stationarity, is_stationary
from ts_stat_tests.utils.data import load_airline

# 1. Load sample time-series data
data = load_airline()

# 2. Run a dispatcher test (defaults to Augmented Dickey-Fuller)
# Returns a tuple of (statistic, pvalue, lags, nobs, critical_values, icbest)
results = stationarity(data, algorithm="adf")
print(f"ADF Statistic: {results[0]:.4f}")
print(f"p-value: {results[1]:.4f}")

# 3. Use a Boolean check for quick assessment
# Returns a dictionary with the Boolean result and metadata
check = is_stationary(data, algorithm="adf", alpha=0.05)
if check["result"]:
    print("The time series is stationary.")
else:
    print(f"Non-stationary (p-value: {check['pvalue']:.4f})")
```


## 4. Utility Data Loaders

The `ts_stat_tests.utils.data` module provides several functions to generate or load time-series data for testing and benchmarking:

-   `load_airline()`: Classic monthly airline passenger numbers.
-   `load_macrodata()`: US Macroeconomic data.
-   `get_sine_wave()`: Generates a deterministic sine wave.
-   `get_noise_data()`: Generates random Gaussian noise.
-   `get_random_generator()`: Returns a configured NumPy random generator.


## 5. Available Test Categories

The library covers a wide range of statistical properties essential for time-series analysis:

| Category               | Primary Dispatcher           | Boolean Check               |
| :--------------------- | :--------------------------- | :-------------------------- |
| **Stationarity**       | `stationarity()`             | `is_stationary()`           |
| **Normality**          | `normality()`                | `is_normal()`               |
| **Correlation**        | `correlation()`              | `is_correlated()`           |
| **Heteroscedasticity** | `heteroscedasticity()`       | `is_heteroscedastic()`      |
| **Linearity**          | `linearity()`                | `is_linear()`               |
| **Seasonality**        | `seasonality()`              | `is_seasonal()`             |
| **Stability**          | `stability()`, `lumpiness()` | `is_stable()`, `is_lumpy()` |
| **Regularity**         | `regularity()`               | `is_regular()`              |

For detailed documentation on each algorithm and its parameters, please refer to the [API Reference](../code/index.md).


[ts-stat-tests]: https://data-science-extensions.com/toolboxes/ts-stat-tests
[uv]: https://docs.astral.sh/uv/
