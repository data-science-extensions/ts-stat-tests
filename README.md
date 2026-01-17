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


## Motivation

Time Series Analysis has been around for a long time, especially for doing Statistical Testing. Some Python packages are going a long way to make this even easier than it has ever been before. Such as [`sktime`](https://sktime.org/) and [`pycaret`](https://pycaret.org/) and [`pmdarima`](https://www.google.com/search?q=pmdarima) and [`statsmodels`](https://www.statsmodels.org/).

There are some typical Statistical Tests which are accessible in these Python ([Normality], [Stationarity], [Correlation], [Stability], etc). However, there are still some statistical tests which are not yet ported over to Python, but which have been written in R and are quite stable.

Moreover, there is no one single library package for doing time-series statistical tests in Python.

That's exactly what this package aims to achieve.

A single package for doing all the standard time-series statistical tests.


## Tests

Full credit goes to the packages listed in this table.

| Type               | Name                                                                          | Source Package | Source Language | Implemented |
| ------------------ | ----------------------------------------------------------------------------- | -------------- | --------------- | ----------- |
| Correlation        | Auto-Correlation function (ACF)                                               | `statsmodels`  | Python          | ✅           |
| Correlation        | Partial Auto-Correlation function (PACF)                                      | `statsmodels`  | Python          | ✅           |
| Correlation        | Cross-Correlation function (CCF)                                              | `statsmodels`  | Python          | ✅           |
| Correlation        | Ljung-Box test of autocorrelation in residuals (LB)                           | `statsmodels`  | Python          | ✅           |
| Correlation        | Lagrange Multiplier tests for autocorrelation (LM)                            | `statsmodels`  | Python          | ✅           |
| Correlation        | Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation (BGLM) | `statsmodels`  | Python          | ✅           |
| Regularity         | Approximate Entropy                                                           | `antropy`      | python          | ✅           |
| Regularity         | Sample Entropy                                                                | `antropy`      | python          | ✅           |
| Regularity         | Permutation Entropy                                                           | `antropy`      | python          | ✅           |
| Regularity         | Spectral Entropy                                                              | `antropy`      | python          | ✅           |
| Regularity         | SVD Entropy                                                                   | `antropy`      | python          | ✅           |
| Seasonality        | QS                                                                            | `seastests`    | R               | ✅           |
| Seasonality        | Osborn-Chui-Smith-Birchenhall test of seasonality (OCSB)                      | `pmdarima`     | Python          | ✅           |
| Seasonality        | Canova-Hansen test for seasonal differences (CH)                              | `pmdarima`     | Python          | ✅           |
| Seasonality        | Seasonal Strength                                                             | `tsfeatures`   | Python          | ✅           |
| Seasonality        | Trend Strength                                                                | `tsfeatures`   | Python          | ✅           |
| Seasonality        | Spikiness                                                                     | `tsfeatures`   | Python          | ✅           |
| Stability          | Stability                                                                     | `tsfeatures`   | Python          | ✅           |
| Stability          | Lumpiness                                                                     | `tsfeatures`   | Python          | ✅           |
| Stationarity       | Augmented Dickey-Fuller test for stationarity (ADF)                           | `statsmodels`  | Python          | ✅           |
| Stationarity       | Kwiatkowski-Phillips-Schmidt-Shin test for stationarity (KPSS)                | `statsmodels`  | Python          | ✅           |
| Stationarity       | Range unit-root test for stationarity (RUR)                                   | `statsmodels`  | Python          | ✅           |
| Stationarity       | Zivot-Andrews structural-break unit-root test (ZA)                            | `statsmodels`  | Python          | ✅           |
| Stationarity       | Phillips-Peron test for stationarity (PP)                                     | `arch`         | Python          | ✅           |
| Stationarity       | Elliott-Rothenberg-Stock (ERS) de-trended Dickey-Fuller test                  | `arch`         | Python          | ✅           |
| Stationarity       | Variance Ratio (VR) test for a random walk                                    | `arch`         | Python          | ✅           |
| Normality          | Jarque-Bera test of normality (JB)                                            | `statsmodels`  | Python          | ✅           |
| Normality          | Omnibus test for normality (OB)                                               | `statsmodels`  | Python          | ✅           |
| Normality          | Shapiro-Wilk test for normality (SW)                                          | `scipy`        | Python          | ✅           |
| Normality          | D'Agostino & Pearson's test for normality                                     | `scipy`        | Python          | ✅           |
| Normality          | Anderson-Darling test for normality                                           | `scipy`        | Python          | ✅           |
| Linearity          | Harvey Collier test for linearity (HC)                                        | `statsmodels`  | Python          | ✅           |
| Linearity          | Lagrange Multiplier test for linearity (LM)                                   | `statsmodels`  | Python          | ✅           |
| Linearity          | Rainbow test for linearity (RB)                                               | `statsmodels`  | Python          | ✅           |
| Linearity          | Ramsey's RESET test for neglected nonlinearity (RR)                           | `statsmodels`  | Python          | ✅           |
| Heteroscedasticity | Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH)         | `statsmodels`  | Python          | 🔲           |
| Heteroscedasticity | Breusch-Pagan Lagrange Multiplier test for heteroscedasticity (BPL)           | `statsmodels`  | Python          | 🔲           |
| Heteroscedasticity | Goldfeld-Quandt test for homoskedasticity (GQ)                                | `statsmodels`  | Python          | 🔲           |
| Heteroscedasticity | White's Lagrange Multiplier Test for Heteroscedasticity (WLM)                 | `statsmodels`  | Python          | 🔲           |


## Known limitations

- These listed tests is not exhaustive, and there is probably some more that could be added. Therefore, we encourage you to raise issues or pull requests to add more statistical tests to this suite.
- This package does not re-invent any of these tests. It merely calls the underlying packages, and calls the functions which are already written elsewhere.


[Normality]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/normality/
[Stationarity]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/stationarity/
[Correlation]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/correlation/
[Stability]: https://data-science-extensions.com/toolboxes/ts-stat-tests/latest/code/stability/
[badge-license]: https://img.shields.io/pypi/l/ts-stat-tests?logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMCAyMCI+DQogICAgPHBhdGggZmlsbD0id2hpdGUiDQogICAgICAgIGQ9Ik0yLjM4MSAzLjE3NUMyLjM4MSAxLjQyMSAzLjgwMiAwIDUuNTU2IDBoMTEuMTExYy41MjYgMCAuOTUyLjQyNi45NTIuOTUydjE1Ljg3M2MwIC41MjYtLjQyNi45NTMtLjk1Mi45NTNoLTMuMTc1Yy0uNzMzIDAtMS4xOTEtLjc5NC0uODI1LTEuNDI5LjE3LS4yOTQuNDg1LS40NzYuODI1LS40NzZoMi4yMjJ2LTIuNTRINS41NTZjLS45NzggMC0xLjU4OSAxLjA1OS0xLjEgMS45MDUuMDU0LjA5My4xMTguMTc4LjE5My4yNTQuNTEzLjUyNC4yNjcgMS40MDctLjQ0NCAxLjU5LS4zMjkuMDg0LS42NzktLjAxMy0uOTE3LS4yNTctLjU4Mi0uNTkzLS45MDgtMS4zOTEtLjkwNy0yLjIyMlYzLjE3NVptMTMuMzMzLTEuMjdINS41NTZjLS43MDIgMC0xLjI3LjU2OC0xLjI3IDEuMjd2OC41MThjLjQtLjE3NS44MzMtLjI2NSAxLjI3LS4yNjRoMTAuMTU4VjEuOTA1Wk02LjE5MSAxNS41NTZjMC0uMTc2LjE0Mi0uMzE4LjMxNy0uMzE4aDQuNDQ0Yy4xNzYgMCAuMzE4LjE0Mi4zMTguMzE4djQuMTI3YzAgLjI0NC0uMjY1LjM5Ny0uNDc2LjI3NC0uMDExLS4wMDYtLjAyMi0uMDEzLS4wMzItLjAybC0xLjg0MS0xLjM4MWMtLjExMy0uMDg1LS4yNjktLjA4NS0uMzgxIDBsLTEuODQyIDEuMzgxYy0uMTk1LjE0Ni0uNDc2LjAyNi0uNTA1LS4yMTYtLjAwMi0uMDEzLS4wMDItLjAyNi0uMDAyLS4wMzh2LTQuMTI3WiIgLz4NCjwvc3ZnPg==
[badge-downloads]: https://img.shields.io/pypi/dw/ts-stat-tests?label=downloads&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDIwIDIwIj4NCiAgICA8cGF0aCBmaWxsPSJ3aGl0ZSINCiAgICAgICAgZD0iTSAxNy43NzggMTIuMjIyIEwgMTcuNzc4IDE3Ljc3OCBMIDIuMjIyIDE3Ljc3OCBMIDIuMjIyIDEyLjIyMiBMIDAgMTIuMjIyIEwgMCAxNy43NzggQyAwIDE5LjAwNSAwLjk5NSAyMCAyLjIyMiAyMCBMIDE3Ljc3OCAyMCBDIDE5LjAwNSAyMCAyMCAxOS4wMDUgMjAgMTcuNzc4IEwgMjAgMTIuMjIyIEwgMTcuNzc4IDEyLjIyMiBaIE0gMTAgMTUuNTU2IEwgMTUuNTU2IDguODg5IEwgMTEuMTExIDguODg5IEwgMTEuMTExIDAgTCA4Ljg4OSAwIEwgOC44ODkgOC44ODkgTCA0LjQ0NCA4Ljg4OSBMIDEwIDE1LjU1NiBaIiAvPg0KPC9zdmc+
[badge-style]: https://img.shields.io/badge/code_style-black-000000.svg?logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDIwIDIwIj4NCiAgICA8cGF0aCBmaWxsPSJ3aGl0ZSINCiAgICAgICAgZD0ibTExLjc2IDE5LjYxMiA4LjIxNS03LjMwNC03Ljc4My02LjkyYy0uNjM2LS41NjMtMS42MDgtLjUwNy0yLjE3Mi4xMjgtLjU2NS42MzUtLjUwOCAxLjYwOC4xMjggMi4xNzNsNS4xOTggNC42MTktNS42MyA1LjAwMmMtLjYzNS41NjUtLjY5MSAxLjUzNy0uMTI4IDIuMTczLjMwMy4zNDMuNzI2LjUxNyAxLjE1LjUxNy4zNjMgMCAuNzI5LS4xMjggMS4wMjItLjM4OFpNOC44MDYgMTVjLS4zNjMgMC0uNzMtLjEyOC0xLjAyMi0uMzg4TDAgNy42OTIgOC4yMTYuMzg2Yy42MzUtLjU2IDEuNjA3LS41MDYgMi4xNzIuMTI4LjU2NC42MzYuNTA3IDEuNjA5LS4xMjggMi4xNzRMNC42MyA3LjY5Mmw1LjE5OCA0LjYxOWMuNjM2LjU2My42OTMgMS41MzYuMTI4IDIuMTcyLS4zMDQuMzQzLS43MjcuNTE3LTEuMTUuNTE3WiIgLz4NCjwvc3ZnPg==
