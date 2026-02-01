# Detailed Analysis of Algorithms

This page provides a comprehensive overview of the statistical algorithms integrated into `ts-stat-tests`. It details the rationale behind our library selections, provides an exhaustive catalog of implemented tests, and demonstrates advanced usage patterns for power users and developers.


!!! note "TL;DR"

    There are a number of other really good libraries which implement these tests individually:

    - [`pmdarima`][pmdarima]
    - [`statsmodels`][statsmodels]
    - [`arch`][arch]
    - [`tsfeatures`][tsfeatures]
    - [`antropy`][antropy]
    - [`scipy`][scipy]

    These packages all implement the statistical tests in a slightly different way.<br>
    However, no one library contains all of the required tests, all in one place.<br>
    That's precisely why we created `ts-stat-tests` - to provide a unified interface to all of these great libraries, wrapped in a consistent and easy-to-use API.


## 💡 Selection Rationale

The primary goal of `ts-stat-tests` is to provide a single, unified interface to the vast landscape of Python's time-series statistical tools. We have selected specific underlying libraries based on several key criteria:

-   **Reliability**: Using industry-standard libraries like `statsmodels` and `scipy` ensures the mathematical correctness of the implementations.
-   **Popularity**: Libraries with strong community support and active maintenance were prioritised to ensure longevity and compatibility.
-   **Performance**: Where possible, we leverage libraries that utilise Numba or Cython for high-speed computation (e.g., `antropy`, `arch`).
-   **API Consistency**: We wrap diverse APIs (which often vary in return types and parameter naming) into a consistent, predictable structure.
-   **Coverage**: We fill gaps where no single library provides a complete suite of tests for a specific domain (e.g., combining `statsmodels` for stationarity with `arch` for variance ratios).


## 📚 Exhaustive Test Catalog

The following tables detail every test currently implemented or planned for the library.

<style>
    table colgroup col {
        width: auto !important;
    }
    .test-info div div table th:nth-of-type(1) {
        width: 10%;
    }
    .test-info div div table th:nth-of-type(2) {
        width: 50%;
    }
    .test-info div div table th:nth-of-type(3) {
        width: 40%;
    }
</style>


!!! info "Details"

    Legend:

    | icon | description                                                                     |
    | ---- | ------------------------------------------------------------------------------- |
    | ✅    | Already implemented in this package                                             |
    | 🔲    | To be developed and implemented                                                 |
    | ❎    | Will not be implemented as it is covered by a function from a different package |

    === "Test Info"

        <div class="test-info">

        | category           | algorithm                                                                     | library:test                                                                                                                                                         |
        | ------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        | Correlation        | Auto-Correlation function (ACF)                                               | ✅[`statsmodels`][statsmodels]:[`acf()`][statsmodels-acf]<br>❎[`pmdarima`][pmdarima]:[`acf()`][pmdarima-acf]                                                          |
        | Correlation        | Partial Auto-Correlation function (PACF)                                      | ✅[`statsmodels`][statsmodels]:[`pacf()`][statsmodels-pacf]<br>❎[`pmdarima`][pmdarima]:[`pacf()`][pmdarima-pacf]                                                      |
        | Correlation        | Cross-Correlation function (CCF)                                              | ✅[`statsmodels`][statsmodels]:[`ccf()`][statsmodels-ccf]                                                                                                             |
        | Correlation        | Ljung-Box test of autocorrelation in residuals (LB)                           | ✅[`statsmodels`][statsmodels]:[`acorr_ljungbox()`][statsmodels-acorr_ljungbox]                                                                                       |
        | Correlation        | Lagrange Multiplier tests for autocorrelation (LM)                            | ✅[`statsmodels`][statsmodels]:[`acorr_lm()`][statsmodels-acorr_lm]                                                                                                   |
        | Correlation        | Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation (BGLM) | ✅[`statsmodels`][statsmodels]:[`acorr_breusch_godfrey()`][statsmodels-acorr_breusch_godfrey]                                                                         |
        | Regularity         | Approximate Entropy                                                           | ✅[`antropy`][antropy]:[`app_entropy()`][antropy-app_entropy]                                                                                                         |
        | Regularity         | Sample Entropy                                                                | ✅[`antropy`][antropy]:[`sample_entropy()`][antropy-sample_entropy]                                                                                                   |
        | Regularity         | Permutation Entropy                                                           | ✅[`antropy`][antropy]:[`perm_entropy()`][antropy-perm_entropy]                                                                                                       |
        | Regularity         | Spectral Entropy                                                              | ✅[`antropy`][antropy]:[`spectral_entropy()`][antropy-spectral_entropy]                                                                                               |
        | Regularity         | SVD Entropy                                                                   | ✅[`antropy`][antropy]:[`svd_entropy()`][antropy-svd_entropy]                                                                                                         |
        | Seasonality        | QS                                                                            | ✅[`seastests`][seastests]:[`qs()`][seastests-qs]                                                                                                                     |
        | Seasonality        | Osborn-Chui-Smith-Birchenhall test of seasonality (OCSB)                      | ✅[`pmdarima`][pmdarima]:[`OCSBTest()`][pmdarima-ocsbtest]                                                                                                            |
        | Seasonality        | Canova-Hansen test for seasonal differences (CH)                              | ✅[`pmdarima`][pmdarima]:[`CHTest()`][pmdarima-chtest]                                                                                                                |
        | Seasonality        | Seasonal Strength                                                             | ✅[`tsfeatures`][tsfeatures]:[`stl_features()`][tsfeatures-stl_features]                                                                                              |
        | Seasonality        | Trend Strength                                                                | ✅[`tsfeatures`][tsfeatures]:[`stl_features()`][tsfeatures-stl_features]                                                                                              |
        | Seasonality        | Spikiness                                                                     | ✅[`tsfeatures`][tsfeatures]:[`stl_features()`][tsfeatures-stl_features]                                                                                              |
        | Stability          | Stability                                                                     | ✅[`tsfeatures`][tsfeatures]:[`stability()`][tsfeatures-stability]                                                                                                    |
        | Stability          | Lumpiness                                                                     | ✅[`tsfeatures`][tsfeatures]:[`lumpiness()`][tsfeatures-lumpiness]                                                                                                    |
        | Stationarity       | Augmented Dickey-Fuller test for stationarity (ADF)                           | ✅[`statsmodels`][statsmodels]:[`adfuller()`][statsmodels-adfuller]<br>❎[`pmdarima`][pmdarima]:[`ADFTest()`][pmdarima-adftest]<br>❎[`arch`][arch]:[`ADF()`][arch-adf] |
        | Stationarity       | Kwiatkowski-Phillips-Schmidt-Shin test for stationarity (KPSS)                | ✅[`statsmodels`][statsmodels]:[`kpss()`][statsmodels-kpss]<br>❎[`pmdarima`][pmdarima]:[`KPSSTest()`][pmdarima-kpss]<br>❎[`arch`][arch]:[`KPSS()`][arch-kpss]         |
        | Stationarity       | Range unit-root test for stationarity (RUR)                                   | ✅[`statsmodels`][statsmodels]:[`range_unit_root_test()`][statsmodels-range_unit_root_test]                                                                           |
        | Stationarity       | Zivot-Andrews structural-break unit-root test (ZA)                            | ✅[`statsmodels`][statsmodels]:[`zivot_andrews()`][statsmodels-zivot_andrews]<br>❎[`arch`][arch]:[`ZivotAndrews()`][arch-zivotandrews]                                |
        | Stationarity       | Phillips-Peron test for stationarity (PP)                                     | ✅[`pmdarima`][pmdarima]:[`PPTest()`][pmdarima-pptest]<br>❎[`arch`][arch]:[`PhillipsPerron()`][arch-phillipsperron]                                                   |
        | Stationarity       | Elliott-Rothenberg-Stock (ERS) de-trended Dickey-Fuller test                  | ✅[`arch`][arch]:[`DFGLS()`][arch-dfgls]                                                                                                                              |
        | Stationarity       | Variance Ratio (VR) test for a random walk                                    | ✅[`arch`][arch]:[`VarianceRatio()`][arch-varianceratio]                                                                                                              |
        | Normality          | Jarque-Bera test of normality (JB)                                            | ✅[`statsmodels`][statsmodels]:[`jarque_bera()`][statsmodels-jarque_bera]                                                                                             |
        | Normality          | Omnibus test for normality (OB)                                               | ✅[`statsmodels`][statsmodels]:[`omni_normtest()`][statsmodels-omni_normtest]                                                                                         |
        | Normality          | Shapiro-Wilk test for normality (SW)                                          | ✅[`scipy`][scipy]:[`shapiro()`][scipy-shapiro]                                                                                                                       |
        | Normality          | D'Agostino & Pearson's test for normality (DP)                                | ✅[`scipy`][scipy]:[`normaltest()`][scipy-normaltest]                                                                                                                 |
        | Normality          | Anderson-Darling test for normality (AD)                                      | ✅[`scipy`][scipy]:[`anderson()`][scipy-anderson]                                                                                                                     |
        | Linearity          | Harvey Collier test for linearity (HC)                                        | ✅[`statsmodels`][statsmodels]:[`linear_harvey_collier()`][statsmodels-linear_harvey_collier]                                                                         |
        | Linearity          | Lagrange Multiplier test for linearity (LM)                                   | ✅[`statsmodels`][statsmodels]:[`linear_lm()`][statsmodels-linear_lm]                                                                                                 |
        | Linearity          | Rainbow test for linearity (RB)                                               | ✅[`statsmodels`][statsmodels]:[`linear_rainbow()`][statsmodels-linear_rainbow]                                                                                       |
        | Linearity          | Ramsey's RESET test for neglected nonlinearity (RR)                           | ✅[`statsmodels`][statsmodels]:[`linear_reset()`][statsmodels-linear_reset]                                                                                           |
        | Heteroscedasticity | Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH)         | ✅[`statsmodels`][statsmodels]:[`het_arch()`][statsmodels-het_arch]                                                                                                   |
        | Heteroscedasticity | Breusch-Pagan Lagrange Multiplier test for heteroscedasticity (BPL)           | ✅[`statsmodels`][statsmodels]:[`het_breuschpagan()`][statsmodels-het_breuschpagan]                                                                                   |
        | Heteroscedasticity | Goldfeld-Quandt test for homoskedasticity (GQ)                                | ✅[`statsmodels`][statsmodels]:[`het_goldfeldquandt()`][statsmodels-het_goldfeldquandt]                                                                               |
        | Heteroscedasticity | White's Lagrange Multiplier Test for Heteroscedasticity (WLM)                 | ✅[`statsmodels`][statsmodels]:[`het_white()`][statsmodels-het_white]                                                                                                 |

        </div>

    === "Python Import"

        | test | library:import                                                                                                                                           |
        | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
        | ADF  | pmdarima: `from pmdarima.arima import ADFTest`<br>statsmodels: `from statsmodels.tsa.stattools import adfuller`<br>arch: `from arch.unitroot import ADF` |
        | KPSS | pmdarima: `from pmdarima.arima import KPSSTest`<br>statsmodels: `from statsmodels.tsa.stattools import kpss`<br>arch: `from arch.unitroot import KPSS`   |
        | PP   | pmdarima: `from pmdarima.arima import PPTest`<br>arch: `from arch.unitroot import PhillipsPerron`                                                        |
        | RUR  | statsmodels: `from statsmodels.tsa.stattools import range_unit_root_test`                                                                                |
        | ZA   | statsmodels: `from statsmodels.tsa.stattools import zivot_andrews`<br> arch: `from arch.unitroot import ZivotAndrews`                                    |
        | OCSB | pmdarima: `from pmdarima.arima import OCSBTest`                                                                                                          |
        | CH   | pmdarima: `from pmdarima.arima import CHTest`                                                                                                            |
        | ACF  | pmdarima: `from pmdarima.utils import acf`<br>statsmodels: `from statsmodels.tsa.stattools import acf`                                                   |
        | PACF | pmdarima: `from pmdarima.utils import pacf`<br>statsmodels: `from statsmodels.tsa.stattools import pacf`                                                 |
        | CCF  | statsmodels: `from statsmodels.tsa.stattools import ccf`                                                                                                 |
        | ALB  | statsmodels: `from statsmodels.stats.diagnostic import acorr_ljungbox`                                                                                   |
        | ALM  | statsmodels: `from statsmodels.stats.diagnostic import acorr_lm`                                                                                         |
        | ABG  | statsmodels: `from statsmodels.stats.diagnostic import acorr_breusch_godfrey`                                                                            |
        | JB   | statsmodels: `from statsmodels.stats.stattools import jarque_bera`                                                                                       |
        | OB   | statsmodels: `from statsmodels.stats.stattools import omni_normtest`                                                                                     |
        | HC   | statsmodels: `from statsmodels.stats.diagnostic import linear_harvey_collier`                                                                            |
        | LM   | statsmodels: `from statsmodels.stats.diagnostic import linear_lm`                                                                                        |
        | RB   | statsmodels: `from statsmodels.stats.diagnostic import linear_rainbow`                                                                                   |
        | RR   | statsmodels: `from statsmodels.stats.diagnostic import linear_reset`                                                                                     |
        | ARCH | statsmodels: `from statsmodels.stats.diagnostic import het_arch`                                                                                         |
        | BPL  | statsmodels: `from statsmodels.stats.diagnostic import het_breuschpagan`                                                                                 |
        | GQ   | statsmodels: `from statsmodels.stats.diagnostic import het_goldfeldquandt`                                                                               |
        | WLM  | statsmodels: `from statsmodels.stats.diagnostic import het_white`                                                                                        |


## 🚀 Advanced Usage Patterns

While the high-level dispatchers are convenient, direct access to the underlying `algorithms` allows for fine-grained control over the statistical testing process.


### Direct Algorithm Access

Importing directly from the `.algorithms` submodule grants access to full parameter sets and raw results stores from the underlying libraries.

```py {.py .python linenums="1" title="Advanced Stationarity Example"}
from ts_stat_tests.stationarity.algorithms import adf
from ts_stat_tests.utils.data import load_airline

data = load_airline().values

# Advanced ADF usage:
# - No constant/trend ('n')
# - Fixed lag length of 12
# - Return the full ResultsStore object
stat, pval, crit, resstore = adf(
    data, regression="n", autolag=None, maxlag=12, store=True
)

print(f"Test Statistic: {stat}")
print(f"Regression Summary: \n{resstore.resols.summary()}")
```


### Handling Complex Returns

Many underlying algorithms return different shapes depending on their parameters. We use Python's `@overload` functionality to ensure type safety even with these dynamic returns.

-   **Standard Return**: Usually a tuple of primary statistics (e.g., `(stat, pvalue, ...)`).
-   **Stored Results**: Setting `store=True` often appends a library-specific result object (like `ResultsStore`) to the return tuple.


## 🛠️ Developer Information

For those contributing to `ts-stat-tests`, maintaining the quality and consistency of the internal algorithms is paramount.


### Strict Normalisation

Every internal algorithm wrapper MUST normalise its output. We do not expose raw type inconsistencies from third-party libraries to the user. All outputs should be cast to standard `numpy` or `float`/`int` types before being returned.


### Documentation Requirements (DFC)

All algorithm docstrings must adhere to the **Docstring Format Checker (DFC)** standards using Google style. Specifically, they must include:

1.  `!!! note "Summary"`: High-level purpose.
2.  `???+ abstract "Details"`: Theoretical background.
3.  `??? example "Examples"`: Functional `pycon` examples.
4.  `??? equation "Calculation"`: LaTeX formatted math.
5.  `??? question "References"`: Academic sources.


### Type Hinting and Overloads

We aim to avoid the use of `Any` wherever possible. All parameters and return values must be explicitly typed. Because many statistical functions have conditional return types based on Boolean flags (like `store`), you must use `@overload` to define every possible return signature.


## 📖 References and External Documentation

For deeper dives into the underlying implementations, we recommend consulting the official documentation of our core dependencies:

-   [Statsmodels Documentation][statsmodels]
-   [Arch Documentation][arch]
-   [Pmdarima Documentation][pmdarima]
-   [Antropy Documentation][antropy]
-   [Scipy Stats Documentation][scipy]

[statsmodels]: https://www.statsmodels.org
[statsmodels-adfuller]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
[statsmodels-kpss]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
[statsmodels-range_unit_root_test]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html
[statsmodels-zivot_andrews]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html
[statsmodels-acf]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
[statsmodels-pacf]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html
[statsmodels-ccf]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html
[statsmodels-acorr_ljungbox]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
[statsmodels-acorr_lm]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html
[statsmodels-acorr_breusch_godfrey]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html
[statsmodels-jarque_bera]: https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.jarque_bera.html
[statsmodels-omni_normtest]: https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.omni_normtest.html
[statsmodels-linear_harvey_collier]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_harvey_collier.html
[statsmodels-linear_lm]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_lm.html
[statsmodels-linear_rainbow]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_rainbow.html
[statsmodels-linear_reset]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_reset.html
[statsmodels-het_arch]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html
[statsmodels-het_breuschpagan]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html
[statsmodels-het_goldfeldquandt]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_goldfeldquandt.html
[statsmodels-het_white]: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_white.html
[pmdarima]: https://alkaline-ml.com/pmdarima
[pmdarima-adftest]: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ADFTest.html
[pmdarima-kpss]: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.KPSSTest.html
[pmdarima-pptest]: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html
[pmdarima-ocsbtest]: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.html
[pmdarima-chtest]: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html
[pmdarima-acf]: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.acf.html
[pmdarima-pacf]: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.pacf.html
[arch]: https://arch.readthedocs.io
[arch-adf]: https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.ADF.html
[arch-kpss]: https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.KPSS.html
[arch-phillipsperron]: https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.PhillipsPerron.html
[arch-zivotandrews]: https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.ZivotAndrews.html
[arch-dfgls]: https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html
[arch-varianceratio]: https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html
[seastests]: https://www.rdocumentation.org/packages/seastests
[seastests-qs]: https://www.rdocumentation.org/packages/seastests/versions/0.14.2/topics/qs
[tsfeatures]: https://github.com/Nixtla/tsfeatures
[tsfeatures-stl_features]: https://github.com/Nixtla/tsfeatures/blob/master/tsfeatures/tsfeatures.py#L641
[tsfeatures-stability]: https://github.com/Nixtla/tsfeatures/blob/master/tsfeatures/tsfeatures.py#L608
[tsfeatures-lumpiness]: https://github.com/Nixtla/tsfeatures/blob/master/tsfeatures/tsfeatures.py#L445
[antropy]: https://raphaelvallat.com/antropy
[antropy-app_entropy]: https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html
[antropy-sample_entropy]: https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html
[antropy-perm_entropy]: https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html
[antropy-spectral_entropy]: https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html
[antropy-svd_entropy]: https://raphaelvallat.com/antropy/build/html/generated/antropy.svd_entropy.html
[scipy]: https://docs.scipy.org/
[scipy-shapiro]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
[scipy-normaltest]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
[scipy-anderson]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
