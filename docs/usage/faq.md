# Frequently Asked Questions (FAQ)

This page addresses common questions regarding the usage, design, and technical implementation of the `ts-stat-tests` library.


---


## 🚀 General Usage

### 1. What is the difference between "Dispatcher", "Boolean", and "Algorithm" functions?

The library provides three primary ways to interact with statistical tests:

1.  **Dispatchers (e.g., [`stationarity()`][ts_stat_tests.stationarity.tests.stationarity])**: These function as unified entry points. You pass an `algorithm` string (like `"adf"` or `"kpss"`), and the dispatcher handles the mapping to the underlying implementation. They return a tuple containing the raw results (statistic, p-value, etc.).
2.  **Boolean Checkers (e.g., [`is_stationary()`][ts_stat_tests.stationarity.tests.is_stationary])**: These are high-level convenience functions that interpret the p-values against a significance level ($\alpha$) and return a dictionary with a simple `True`/`False` result.
3.  **Specific Algorithms (e.g., [`adf()`][ts_stat_tests.stationarity.algorithms.adf])**: These are the direct implementations of each test. They offer the most granular control, allowing you to pass complex parameters that might not be exposed through the dispatcher. They are ideal for power users who need to replicate exact results from other software or libraries.


### 2. Why use `ts-stat-tests` instead of calling the underlying packages directly, like `statsmodels` or `arch`?

While you can certainly call those libraries directly, `ts-stat-tests` offers several advantages:

-   **Consistent API**: You don't need to remember if a library expects a 1D array, a DataFrame, or specific keyword arguments. The dispatchers normalise these inputs.
-   **Standardised Outputs**: Results are returned in a consistent format (usually `tuple` or `dict`), making it easier to swap algorithms without changing your downstream processing code.
-   **Batteries Included**: We implement logic for $H_0$ interpretation (which varies between tests like ADF and KPSS) so you don't have to.


### 3. What is the `ResultsStore` object?

Many dispatchers return a `ResultsStore` object (from `statsmodels.stats.diagnostic`). This is a container that holds the full results of the statistical test, including any intermediate calculations or ancillary statistics that might not be part of the primary return tuple.


---


## 💡 Statistical Logic


### 4. How is the Null Hypothesis ($H_0$) handled?

In Boolean functions like [`is_stationary()`][ts_stat_tests.stationarity.tests.is_stationary], we automatically handle the difference in Null Hypotheses:

-   **Unit Root Tests (ADF, PP, ZA, etc.)**: $H_0$ is that the series has a unit root (is non-stationary). We return `True` if `p-value < alpha`.
-   **Stationarity Tests (KPSS)**: $H_0$ is that the series is stationary. We return `True` if `p-value > alpha`.


### 5. Why are some p-values "capped"?

Some underlying libraries (like `statsmodels` for the KPSS test) use interpolation tables for p-values. If the test statistic falls outside the table's range, the p-value is reported as the boundary value (e.g., `0.1` or `0.01`) and a warning may be issued.


---


## 📦 Technical & Installation


### 6. Why is `numpy` pinned to version `< 2.4.0`?

`ts-stat-tests` depends on `numba` for certain high-performance calculations (like heavy regularity or entropy tests). As of the current release, `numba` is not yet fully compatible with `numpy` 2.4+. To ensure stability and prevent LLVM-related crashes, we cap the NumPy version at 2.3.x.


### 7. Why are there `@typechecked` decorators on every function?

We believe in "Fail Fast" development. By using `typeguard`, we ensure that any data passed to the library matches the expected type (e.g., `ArrayLike`). This prevents cryptic errors deep inside mathematical calculations and ensures the library remains reliable in production pipelines.


### 8. Why does the library use Australian English?

The maintainers of this project use Australian English as the primary language for documentation and internal naming (e.g., `normalise` instead of `normalize`). We aim for consistency across the entire codebase.


---


## 🛠️ Troubleshooting


### 9. I'm getting a `ValueError: Invalid 'algorithm': ...`. What should I do?

Our dispatchers use a standardised error generator. The error message will typically list all valid options for the parameter you provided. For example:

```txt {.txt .text linenums="1" title="Example Error Message"}
ValueError: Invalid 'algorithm': 'invalid_name'. Options: ['adf', 'kpss', 'pp', 'za', ...]
```

Check the [Algorithms Guide](algorithms.md) for a full list of supported tests and their aliases.


### 10. How can I get sample data to test the library?

The `ts_stat_tests.utils.data` module contains several built-in datasets and generators:

```py {.py .python linenums="1" title="Data Loading Example"}
from ts_stat_tests.utils.data import data_airline, get_sine_wave

# Load a classic dataset
df = data_airline

# Generate synthetic data
x = get_sine_wave()
```
