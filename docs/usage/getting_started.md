# Getting Started

Welcome to `ts-stat-tests`, a comprehensive Python library designed for rigorous time-series statistical testing. This guide provides an overview of the library's core architecture and demonstrates how to perform your first statistical tests.


## 📦 1. Installation

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


## 💡 2. Core Concepts

The library is structured into functional modules (e.g., `stationarity`, `normality`, `correlation`), each offering three levels of interaction:

1.  **Dispatchers**: Versatile entry points (e.g., `stationarity()`) that wrap multiple specific algorithms under a single interface.
2.  **Boolean Checks**: Convenience functions (e.g., `is_stationary()`) that return a dictionary with a Boolean result based on a significance level (`alpha`).
3.  **Specific Algorithms**: Direct access to underlying implementations (e.g., `adf()`, `kpss()`) for advanced configuration.


## 🚀 3. Quickstart Example

Let's perform a stationarity test on some sample airline data.

```py {.py .python linenums="1" title="Stationarity Example"}
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


## 🛠️ 4. Utility Data Loaders

The `ts_stat_tests.utils.data` module provides several functions to generate or load time-series data for testing and benchmarking:

-   [`load_airline()`][ts_stat_tests.utils.data.load_airline]: Classic monthly airline passenger numbers.
-   [`load_macrodata()`][ts_stat_tests.utils.data.load_macrodata]: US Macroeconomic data.
-   [`get_sine_wave()`][ts_stat_tests.utils.data.get_sine_wave]: Generates a deterministic sine wave.
-   [`get_noise_data()`][ts_stat_tests.utils.data.get_noise_data]: Generates random Gaussian noise.
-   [`get_random_generator()`][ts_stat_tests.utils.data.get_random_generator]: Returns a configured NumPy random generator.


## 📚 5. Available Test Categories

The library covers a wide range of statistical properties essential for time-series analysis:

| Category                                            | Primary Dispatcher                                                                                                 | Boolean Check                                                                                                    | Available Algorithms                                                                                                                                                                                                                                                                                                                                                                     |
| :-------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Correlation](../../code/correlation)               | [`correlation()`][ts_stat_tests.correlation.tests.correlation]                                                     | [`is_correlated()`][ts_stat_tests.correlation.tests.is_correlated]                                               | [`acf()`][ts_stat_tests.correlation.algorithms.acf], [`pacf()`][ts_stat_tests.correlation.algorithms.pacf], [`ccf()`][ts_stat_tests.correlation.algorithms.ccf], [`lb()`][ts_stat_tests.correlation.algorithms.lb], [`lm()`][ts_stat_tests.correlation.algorithms.lm], [`bglm()`][ts_stat_tests.correlation.algorithms.bglm]                                                             |
| [Regularity](../../code/regularity)                 | [`regularity()`][ts_stat_tests.regularity.tests.regularity], [`entropy()`][ts_stat_tests.regularity.tests.entropy] | [`is_regular()`][ts_stat_tests.regularity.tests.is_regular]                                                      | [`approx_entropy()`][ts_stat_tests.regularity.algorithms.approx_entropy], [`sample_entropy()`][ts_stat_tests.regularity.algorithms.sample_entropy], [`permutation_entropy()`][ts_stat_tests.regularity.algorithms.permutation_entropy], [`spectral_entropy()`][ts_stat_tests.regularity.algorithms.spectral_entropy], [`svd_entropy()`][ts_stat_tests.regularity.algorithms.svd_entropy] |
| [Normality](../../code/normality)                   | [`normality()`][ts_stat_tests.normality.tests.normality]                                                           | [`is_normal()`][ts_stat_tests.normality.tests.is_normal]                                                         | [`jb()`][ts_stat_tests.normality.algorithms.jb], [`ob()`][ts_stat_tests.normality.algorithms.ob], [`sw()`][ts_stat_tests.normality.algorithms.sw], [`dp()`][ts_stat_tests.normality.algorithms.dp], [`ad()`][ts_stat_tests.normality.algorithms.ad]                                                                                                                                      |
| [Stationarity](../../code/stationarity)             | [`stationarity()`][ts_stat_tests.stationarity.tests.stationarity]                                                  | [`is_stationary()`][ts_stat_tests.stationarity.tests.is_stationary]                                              | [`adf()`][ts_stat_tests.stationarity.algorithms.adf], [`kpss()`][ts_stat_tests.stationarity.algorithms.kpss], [`rur()`][ts_stat_tests.stationarity.algorithms.rur], [`za()`][ts_stat_tests.stationarity.algorithms.za], [`pp()`][ts_stat_tests.stationarity.algorithms.pp], [`ers()`][ts_stat_tests.stationarity.algorithms.ers], [`vr()`][ts_stat_tests.stationarity.algorithms.vr]     |
| [Seasonality](../../code/seasonality)               | [`seasonality()`][ts_stat_tests.seasonality.tests.seasonality]                                                     | [`is_seasonal()`][ts_stat_tests.seasonality.tests.is_seasonal]                                                   | [`qs()`][ts_stat_tests.seasonality.algorithms.qs], [`ocsb()`][ts_stat_tests.seasonality.algorithms.ocsb], [`ch()`][ts_stat_tests.seasonality.algorithms.ch], [`seasonal_strength()`][ts_stat_tests.seasonality.algorithms.seasonal_strength], [`trend_strength()`][ts_stat_tests.seasonality.algorithms.trend_strength], [`spikiness()`][ts_stat_tests.seasonality.algorithms.spikiness] |
| [Stability](../../code/stability)                   | [`stability()`][ts_stat_tests.stability.algorithms.stability]                                                      | [`is_stable()`][ts_stat_tests.stability.tests.is_stable], [`is_lumpy()`][ts_stat_tests.stability.tests.is_lumpy] | [`stability()`][ts_stat_tests.stability.algorithms.stability], [`lumpiness()`][ts_stat_tests.stability.algorithms.lumpiness]                                                                                                                                                                                                                                                             |
| [Linearity](../../code/linearity)                   | [`linearity()`][ts_stat_tests.linearity.tests.linearity]                                                           | [`is_linear()`][ts_stat_tests.linearity.tests.is_linear]                                                         | [`hc()`][ts_stat_tests.linearity.algorithms.hc], [`lm()`][ts_stat_tests.linearity.algorithms.lm], [`rb()`][ts_stat_tests.linearity.algorithms.rb], [`rr()`][ts_stat_tests.linearity.algorithms.rr]                                                                                                                                                                                       |
| [Heteroscedasticity](../../code/heteroscedasticity) | [`heteroscedasticity()`][ts_stat_tests.heteroscedasticity.tests.heteroscedasticity]                                | [`is_heteroscedastic()`][ts_stat_tests.heteroscedasticity.tests.is_heteroscedastic]                              | [`arch()`][ts_stat_tests.heteroscedasticity.algorithms.arch], [`bpl()`][ts_stat_tests.heteroscedasticity.algorithms.bpl], [`gq()`][ts_stat_tests.heteroscedasticity.algorithms.gq], [`wlm()`][ts_stat_tests.heteroscedasticity.algorithms.wlm]                                                                                                                                           |

For detailed documentation on each algorithm and its parameters, please refer to the [API Reference](../code/index.md).


[ts-stat-tests]: https://data-science-extensions.com/toolboxes/ts-stat-tests
[uv]: https://docs.astral.sh/uv/
