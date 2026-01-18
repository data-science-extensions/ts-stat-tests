<h1 align="center"><u><code>ts-stat-tests</code></u></h1>

<p align="center">
<a href="https://github.com/data-science-extensions/ts-stat-tests/releases">
    <img src="https://img.shields.io/github/v/release/data-science-extensions/ts-stat-tests?logo=github" alt="github-release"></a>
<a href="https://pypi.org/project/ts-stat-tests">
    <img src="https://img.shields.io/pypi/implementation/ts-stat-tests?logo=pypi&logoColor=ffde57" alt="implementation"></a>
<a href="https://pypi.org/project/ts-stat-tests">
    <img src="https://img.shields.io/pypi/v/ts-stat-tests?label=version&logo=python&logoColor=ffde57&color=blue" alt="version"></a>
<a href="https://pypi.org/project/ts-stat-tests">
    <img src="https://img.shields.io/pypi/pyversions/ts-stat-tests?logo=python&logoColor=ffde57" alt="python-versions"></a>
<br>
<a href="https://github.com/data-science-extensions/ts-stat-tests/actions/workflows/ci.yml">
    <img src="https://img.shields.io/static/v1?label=os&message=ubuntu+|+macos+|+windows&color=blue&logo=ubuntu&logoColor=green" alt="os"></a>
<a href="https://pypi.org/project/ts-stat-tests">
    <img src="https://img.shields.io/pypi/status/ts-stat-tests?color=green" alt="pypi-status"></a>
<a href="https://pypi.org/project/ts-stat-tests">
    <img src="https://img.shields.io/pypi/format/ts-stat-tests?color=green" alt="pypi-format"></a>
<a href="https://github.com/data-science-extensions/ts-stat-tests/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/data-science-extensions/ts-stat-tests?color=green" alt="github-license"></a>
<a href="https://piptrends.com/package/ts-stat-tests">
    <img src="https://img.shields.io/pypi/dm/ts-stat-tests?color=green" alt="pypi-downloads"></a>
<a href="https://codecov.io/gh/data-science-extensions/ts-stat-tests">
    <img src="https://codecov.io/gh/data-science-extensions/ts-stat-tests/graph/badge.svg" alt="codecov-repo"></a>
<a href="https://github.com/psf/black">
    <img src="https://img.shields.io/static/v1?label=style&message=black&color=black&logo=windows-terminal&logoColor=white" alt="style"></a>
<br>
<a href="https://github.com/data-science-extensions/ts-stat-tests">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat" alt="contributions"></a>
<br>
<a href="https://github.com/data-science-extensions/ts-stat-tests/actions/workflows/ci.yml">
    <img src="https://github.com/data-science-extensions/ts-stat-tests/actions/workflows/ci.yml/badge.svg?event=pull_request" alt="CI"></a>
<a href="https://github.com/data-science-extensions/ts-stat-tests/actions/workflows/cd.yml">
    <img src="https://github.com/data-science-extensions/ts-stat-tests/actions/workflows/cd.yml/badge.svg?event=release" alt="CD"></a>
</p>


## 📚 Overview

**`ts-stat-tests`** is a comprehensive, production-ready Python suite for time-series statistical testing. It provides a unified, highly-consistent interface to established tests from [`statsmodels`][statsmodels], [`scipy`][scipy], [`arch`][arch], [`antropy`][antropy], and more.

Whether you are performing exploratory data analysis, validating model assumptions, or building automated forecasting pipelines, `ts-stat-tests` simplifies the process by normalising inputs, outputs, and hypothesis testing logic into a single, cohesive framework.


## 💡 Motivation

Time Series Analysis has been around for a long time, especially for doing Statistical Testing. Some Python packages are going a long way to make this even easier than it has ever been before. Such as [`sktime`][sktime] and [`pycaret`][pycaret] and [`pmdarima`][pmdarima] and [`statsmodels`][statsmodels].

There are some typical Statistical Tests which are accessible in these Python ([Normality], [Stationarity], [Correlation], [Stability], etc). However, there are still some statistical tests which are not yet ported over to Python, but which have been written in R and are quite stable.

Moreover, there is no one single library package for doing time-series statistical tests in Python.

That's exactly what this package aims to achieve.

A single package for doing all the standard time-series statistical tests.


## 🚀 Key Features

- **Unified Dispatcher API**: Every test category (e.g., Stationarity, Normality) has a single entry point. No more hunting through different libraries for different algorithms.
- **Smart Boolean Checkers**: Every module includes `is_<test>()` functions (e.g., `is_stationary()`) that return clear Boolean results based on significance levels ($\alpha$).
- **Strictly Typed & Verified**: Full compliance with `Pyright` (strict mode) and `@typechecked` decorators for robust runtime reliability.
- **Comprehensive Documentation**: Every algorithm is documented with its mathematical foundation ($\LaTeX$), usage examples, and direct links to source implementation papers.
- **Production Quality**: Maintains a 10.00/10 Pylint score and 100% path coverage across the entire codebase.
- **Simplified Results**: Standardised `ResultsStore` and dictionary outputs make it easy to integrate into larger data pipelines.


## 📦 Installation

Install the package via `pip`:

```bash
pip install ts-stat-tests
```

Or using `uv` (recommended):

```bash
uv add ts-stat-tests
```


## 🛠️ Quick Start

Using `ts-stat-tests` is designed to be intuitive. You can use the high-level dispatcher or the boolean checkers.


### Using the Boolean Checker

Perfect for automated pipelines where you need to branch logic based on statistical properties.

```python {.py .python linenums="1" title="Python"}
import numpy as np
from ts_stat_tests.stationarity import is_stationary

# Create some random data
data = np.random.normal(0, 1, 100)

# Check for stationarity using the ADF test
result = is_stationary(data, algorithm="adf", alpha=0.05)

if result["result"]:
    print(f"Data is stationary (p-value: {result['pvalue']:.4f})")
else:
    print("Data is non-stationary")
```


### Using the Unified Dispatcher

Get the raw statistical results from any supported algorithm.

```python {.py .python linenums="1" title="Python"}
from ts_stat_tests.normality import normality

# Run multiple normality tests via the same interface
jb_stat, jb_pvalue = normality(data, algorithm="jarque_bera")
sw_stat, sw_pvalue = normality(data, algorithm="shapiro_wilk")
```


## 📊 Supported Tests

Full credit goes to the packages listed in this table.

| Type               | Name                                                                          | Source Package | Source Language |
| ------------------ | ----------------------------------------------------------------------------- | -------------- | --------------- |
|                    |                                                                               |                |                 |
| Correlation        | Auto-Correlation function (ACF)                                               | `statsmodels`  | Python          |
| Correlation        | Partial Auto-Correlation function (PACF)                                      | `statsmodels`  | Python          |
| Correlation        | Cross-Correlation function (CCF)                                              | `statsmodels`  | Python          |
| Correlation        | Ljung-Box test of autocorrelation in residuals (LB)                           | `statsmodels`  | Python          |
| Correlation        | Lagrange Multiplier tests for autocorrelation (LM)                            | `statsmodels`  | Python          |
| Correlation        | Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation (BGLM) | `statsmodels`  | Python          |
|                    |                                                                               |                |                 |
| Regularity         | Approximate Entropy                                                           | `antropy`      | python          |
| Regularity         | Sample Entropy                                                                | `antropy`      | python          |
| Regularity         | Permutation Entropy                                                           | `antropy`      | python          |
| Regularity         | Spectral Entropy                                                              | `antropy`      | python          |
| Regularity         | SVD Entropy                                                                   | `antropy`      | python          |
| Seasonality        | QS                                                                            | `seastests`    | R               |
| Seasonality        | Osborn-Chui-Smith-Birchenhall test of seasonality (OCSB)                      | `pmdarima`     | Python          |
| Seasonality        | Canova-Hansen test for seasonal differences (CH)                              | `pmdarima`     | Python          |
| Seasonality        | Seasonal Strength                                                             | `tsfeatures`   | Python          |
| Seasonality        | Trend Strength                                                                | `tsfeatures`   | Python          |
| Seasonality        | Spikiness                                                                     | `tsfeatures`   | Python          |
|                    |                                                                               |                |                 |
| Stability          | Stability                                                                     | `tsfeatures`   | Python          |
| Stability          | Lumpiness                                                                     | `tsfeatures`   | Python          |
| Stationarity       | Augmented Dickey-Fuller test for stationarity (ADF)                           | `statsmodels`  | Python          |
| Stationarity       | Kwiatkowski-Phillips-Schmidt-Shin test for stationarity (KPSS)                | `statsmodels`  | Python          |
| Stationarity       | Range unit-root test for stationarity (RUR)                                   | `statsmodels`  | Python          |
| Stationarity       | Zivot-Andrews structural-break unit-root test (ZA)                            | `statsmodels`  | Python          |
| Stationarity       | Phillips-Peron test for stationarity (PP)                                     | `arch`         | Python          |
| Stationarity       | Elliott-Rothenberg-Stock (ERS) de-trended Dickey-Fuller test                  | `arch`         | Python          |
| Stationarity       | Variance Ratio (VR) test for a random walk                                    | `arch`         | Python          |
|                    |                                                                               |                |                 |
| Normality          | Jarque-Bera test of normality (JB)                                            | `statsmodels`  | Python          |
| Normality          | Omnibus test for normality (OB)                                               | `statsmodels`  | Python          |
| Normality          | Shapiro-Wilk test for normality (SW)                                          | `scipy`        | Python          |
| Normality          | D'Agostino & Pearson's test for normality                                     | `scipy`        | Python          |
| Normality          | Anderson-Darling test for normality                                           | `scipy`        | Python          |
| Linearity          | Harvey Collier test for linearity (HC)                                        | `statsmodels`  | Python          |
| Linearity          | Lagrange Multiplier test for linearity (LM)                                   | `statsmodels`  | Python          |
| Linearity          | Rainbow test for linearity (RB)                                               | `statsmodels`  | Python          |
| Linearity          | Ramsey's RESET test for neglected nonlinearity (RR)                           | `statsmodels`  | Python          |
|                    |                                                                               |                |                 |
| Heteroscedasticity | Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH)         | `statsmodels`  | Python          |
| Heteroscedasticity | Breusch-Pagan Lagrange Multiplier test for heteroscedasticity (BPL)           | `statsmodels`  | Python          |
| Heteroscedasticity | Goldfeld-Quandt test for homoskedasticity (GQ)                                | `statsmodels`  | Python          |
| Heteroscedasticity | White's Lagrange Multiplier Test for Heteroscedasticity (WLM)                 | `statsmodels`  | Python          |


## 🛡️ Quality Assurance

We take code quality seriously. Every commit is verified against:

1. **Code Style**: All code adheres to [`black`][black] formatting, including all code chunks in docstrings using the [`blacken-docs`][blacken-docs].
2. **Spell Checking**: All documentation and code comments are spell-checked using [`codespell`][codespell].
3. **Type Safety**: All code is type-checked using [`ty`][ty] and [`pyright`][pyright] and guarded during runtime by using the [`typeguard`][typeguard] library.
4. **Import Sorting**: All imports are sorted and managed using [`isort`][isort] and unused imports are removed using [`pycln`][pycln].
5. **Code Quality**: All code is checked for quality using [`pylint`][pylint] maintaining a score of 10/10, and checked for complexity using [`complexipy`][complexipy].
6. **Docstring Quality**: All docstrings are checked for style and completeness using [`docstring-format-checker`][docstring-format-checker].
7. **Unit Testing**: All code is unit-tested using [`pytest`][pytest] achieving 100% code coverage across the entire codebase, and including all examples in all docstrings tested using [`doctest`][doctest].
8. **Build Testing**: The package is built with [`uv`][uv] and the docs are built with [`mkdocs`][mkdocs] to ensure there are no build errors.


## ⚠️ Known limitations

- These listed tests is not exhaustive, and there is probably some more that could be added. Therefore, we encourage you to raise issues or pull requests to add more statistical tests to this suite.
- This package does not re-invent any of these tests. It merely calls the underlying packages, and calls the functions which are already written elsewhere.


## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide][CONTRIBUTING] for details on our development workflow and how to submit pull requests.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE] file for details.


[Normality]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/normality/
[Stationarity]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/stationarity/
[Correlation]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/correlation/
[Stability]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/stability/

[statsmodels]: https://www.statsmodels.org
[scipy]: https://docs.scipy.org/
[arch]: https://arch.readthedocs.io
[antropy]: https://raphaelvallat.com/antropy

[sktime]: https://sktime.org/
[pycaret]: https://pycaret.org/
[pmdarima]: https://alkaline-ml.com/pmdarima/
[statsmodels]: https://www.statsmodels.org/

[black]: https://black.readthedocs.io
[blacken-docs]: https://github.com/adamchainz/blacken-docs
[ty]: https://docs.astral.sh/ty/
[pyright]: https://microsoft.github.io/pyright/
[typeguard]: https://typeguard.readthedocs.io
[isort]: https://pycqa.github.io/isort/
[pycln]: https://hadialqattan.github.io/pycln/
[codespell]: https://github.com/codespell-project/codespell
[pylint]: https://pylint.readthedocs.io/
[complexipy]: https://rohaquinlop.github.io/complexipy/
[docstring-format-checker]: https://data-science-extensions.com/toolboxes/docstring-format-checker/
[pytest]: https://docs.pytest.org/
[doctest]: https://docs.python.org/3/library/doctest.html
[uv]: https://docs.astral.sh/uv/
[mkdocs]: https://www.mkdocs.org/

[LICENSE]: https://github.com/data-science-extensions/ts-stat-tests/blob/main/LICENSE
[CONTRIBUTING]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/usage/contributing/

[badge-license]: https://img.shields.io/pypi/l/ts-stat-tests?logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMCAyMCI+DQogICAgPHBhdGggZmlsbD0id2hpdGUiDQogICAgICAgIGQ9Ik0yLjM4MSAzLjE3NUMyLjM4MSAxLjQyMSAzLjgwMiAwIDUuNTU2IDBoMTEuMTExYy41MjYgMCAuOTUyLjQyNi45NTIuOTUydjE1Ljg3M2MwIC41MjYtLjQyNi45NTMtLjk1Mi45NTNoLTMuMTc1Yy0uNzMzIDAtMS4xOTEtLjc5NC0uODI1LTEuNDI5LjE3LS4yOTQuNDg1LS40NzYuODI1LS40NzZoMi4yMjJ2LTIuNTRINS41NTZjLS45NzggMC0xLjU4OSAxLjA1OS0xLjEgMS45MDUuMDU0LjA5My4xMTguMTc4LjE5My4yNTQuNTEzLjUyNC4yNjcgMS40MDctLjQ0NCAxLjU5LS4zMjkuMDg0LS42NzktLjAxMy0uOTE3LS4yNTctLjU4Mi0uNTkzLS45MDgtMS4zOTEtLjkwNy0yLjIyMlYzLjE3NVptMTMuMzMzLTEuMjdINS41NTZjLS43MDIgMC0xLjI3LjU2OC0xLjI3IDEuMjd2OC41MThjLjQtLjE3NS44MzMtLjI2NSAxLjI3LS4yNjRoMTAuMTU4VjEuOTA1Wk02LjE5MSAxNS41NTZjMC0uMTc2LjE0Mi0uMzE4LjMxNy0uMzE4aDQuNDQ0Yy4xNzYgMCAuMzE4LjE0Mi4zMTguMzE4djQuMTI3YzAgLjI0NC0uMjY1LjM5Ny0uNDc2LjI3NC0uMDExLS4wMDYtLjAyMi0uMDEzLS4wMzItLjAybC0xLjg0MS0xLjM4MWMtLjExMy0uMDg1LS4yNjktLjA4NS0uMzgxIDBsLTEuODQyIDEuMzgxYy0uMTk1LjE0Ni0uNDc2LjAyNi0uNTA1LS4yMTYtLjAwMi0uMDEzLS4wMDItLjAyNi0uMDAyLS4wMzh2LTQuMTI3WiIgLz4NCjwvc3ZnPg==
[badge-downloads]: https://img.shields.io/pypi/dw/ts-stat-tests?label=downloads&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDIwIDIwIj4NCiAgICA8cGF0aCBmaWxsPSJ3aGl0ZSINCiAgICAgICAgZD0iTSAxNy43NzggMTIuMjIyIEwgMTcuNzc4IDE3Ljc3OCBMIDIuMjIyIDE3Ljc3OCBMIDIuMjIyIDEyLjIyMiBMIDAgMTIuMjIyIEwgMCAxNy43NzggQyAwIDE5LjAwNSAwLjk5NSAyMCAyLjIyMiAyMCBMIDE3Ljc3OCAyMCBDIDE5LjAwNSAyMCAyMCAxOS4wMDUgMjAgMTcuNzc4IEwgMjAgMTIuMjIyIEwgMTcuNzc4IDEyLjIyMiBaIE0gMTAgMTUuNTU2IEwgMTUuNTU2IDguODg5IEwgMTEuMTExIDguODg5IEwgMTEuMTExIDAgTCA4Ljg4OSAwIEwgOC44ODkgOC44ODkgTCA0LjQ0NCA4Ljg4OSBMIDEwIDE1LjU1NiBaIiAvPg0KPC9zdmc+
[badge-style]: https://img.shields.io/badge/code_style-black-000000.svg?logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDIwIDIwIj4NCiAgICA8cGF0aCBmaWxsPSJ3aGl0ZSINCiAgICAgICAgZD0ibTExLjc2IDE5LjYxMiA4LjIxNS03LjMwNC03Ljc4My02LjkyYy0uNjM2LS41NjMtMS42MDgtLjUwNy0yLjE3Mi4xMjgtLjU2NS42MzUtLjUwOCAxLjYwOC4xMjggMi4xNzNsNS4xOTggNC42MTktNS42MyA1LjAwMmMtLjYzNS41NjUtLjY5MSAxLjUzNy0uMTI4IDIuMTczLjMwMy4zNDMuNzI2LjUxNyAxLjE1LjUxNy4zNjMgMCAuNzI5LS4xMjggMS4wMjItLjM4OFpNOC44MDYgMTVjLS4zNjMgMC0uNzMtLjEyOC0xLjAyMi0uMzg4TDAgNy42OTIgOC4yMTYuMzg2Yy42MzUtLjU2IDEuNjA3LS41MDYgMi4xNzIuMTI4LjU2NC42MzYuNTA3IDEuNjA5LS4xMjggMi4xNzRMNC42MyA3LjY5Mmw1LjE5OCA0LjYxOWMuNjM2LjU2My42OTMgMS41MzYuMTI4IDIuMTcyLS4zMDQuMzQzLS43MjcuNTE3LTEuMTUuNTE3WiIgLz4NCjwvc3ZnPg==
