# Details about the tests

## Intro

!!! note "TL;DR"

    There are a number of other really good libraries which implements these tests individually:

    - [`pmdarima`][pmdarima]
    - [`statsmodels`][statsmodels]
    - [`arch`][arch]
    - [`tsfeatures`][tsfeatures]
    - [`antropy`][antropy]
    - [`scipy`][scipy]

    These packages all implement the statistical tests in a slightly different way.
    <br>
    However, no one library contains all of the required tests, all in one place.


## Implementation Progress

| module             | algorithms               | tests                 | unit-tests                 |
| ------------------ | ------------------------ | --------------------- | -------------------------- |
| Correlation        | [=6/6   "6/6   =  100%"] | [=2/2  "2/2  = 100%"] | [=19/19  "19/19   = 100%"] |
| Regularity         | [=5/5   "5/5   =  100%"] | [=2/2  "2/2  = 100%"] | [=34/34  "34/34   = 100%"] |
| Seasonality        | [=0/6   "0/6   =    0%"] | [=0/2  "0/2  =   0%"] | [=0/10   "0/10    =   0%"] |
| Stability          | [=0/2   "0/2   =    0%"] | [=0/2  "0/2  =   0%"] | [=0/4    "0/4     =   0%"] |
| Stationarity       | [=0/7   "0/7   =    0%"] | [=0/2  "0/2  =   0%"] | [=0/44   "0/44    =   0%"] |
| Normality          | [=5/5   "5/5   =  100%"] | [=2/2  "2/2  = 100%"] | [=12/12  "12/12   = 100%"] |
| Linearity          | [=0/4   "0/4   =    0%"] | [=0/2  "0/2  =   0%"] | [=0/0    "0/0     =   0%"] |
| Heteroscedasticity | [=0/4   "0/4   =    0%"] | [=0/2  "0/2  =   0%"] | [=0/0    "0/0     =   0%"] |
| **Overall**        | [=16/39 "16/39 =   41%"] | [=6/16 "6/16 =  38%"] | [=65/119 "65/119 =  55%"]  |


## Tests

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
        | Linearity          | Harvey Collier test for linearity (HC)                                        | 🔲[`statsmodels`][statsmodels]:[`linear_harvey_collier()`][statsmodels-linear_harvey_collier]                                                                         |
        | Linearity          | Lagrange Multiplier test for linearity (LM)                                   | 🔲[`statsmodels`][statsmodels]:[`linear_lm()`][statsmodels-linear_lm]                                                                                                 |
        | Linearity          | Rainbow test for linearity (RB)                                               | 🔲[`statsmodels`][statsmodels]:[`linear_rainbow()`][statsmodels-linear_rainbow]                                                                                       |
        | Linearity          | Ramsey's RESET test for neglected nonlinearity (RR)                           | 🔲[`statsmodels`][statsmodels]:[`linear_reset()`][statsmodels-linear_reset]                                                                                           |
        | Heteroscedasticity | Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH)         | 🔲[`statsmodels`][statsmodels]:[`het_arch()`][statsmodels-het_arch]                                                                                                   |
        | Heteroscedasticity | Breusch-Pagan Lagrange Multiplier test for heteroscedasticity (BPL)           | 🔲[`statsmodels`][statsmodels]:[`het_breuschpagan()`][statsmodels-het_breuschpagan]                                                                                   |
        | Heteroscedasticity | Goldfeld-Quandt test for homoskedasticity (GQ)                                | 🔲[`statsmodels`][statsmodels]:[`het_goldfeldquandt()`][statsmodels-het_goldfeldquandt]                                                                               |
        | Heteroscedasticity | White's Lagrange Multiplier Test for Heteroscedasticity (WLM)                 | 🔲[`statsmodels`][statsmodels]:[`het_white()`][statsmodels-het_white]                                                                                                 |
        | Covariance         | ...                                                                           |

        </div>

    === "Python Import"

        | test | library:import                                                                                                                                           |
        | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
        | ADF  | pmdarima: `from pmdarima.arima import ADFTest`<br>statsmodels: `from statsmodels.tsa.stattools import adfuller`<br>arch: `from arch.unitroot import ADF` |
        | KPSS | pmdarima: `from pmdarima.arima import KPSSTest`<br>statsmodels: `from statsmodels.tsa.stattools import kpss`<br>arch: `from arch.unitroot import KPSS`   |
        | PP   | pmdarima: `from pmdarima.arima import PPTest`<br>arch: `from arch.unitroot import PhillipsPerron`                                                        |
        | RUR  | pmdarima: `from statsmodels.tsa.stattools import range_unit_root_test`                                                                                   |
        | ZA   | pmdarima: `from statsmodels.tsa.stattools import zivot_andrews`<br> arch: `from arch.unitroot import ZivotAndrews`                                       |
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


<!--
| category     | algorithm                         | import script                         | url                                                                                                          |
| ------------ | --------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Stationarity | Augmented Dickey-Fuller           | `from pmdarima.arima import ADFTest`  | [ADF](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ADFTest.html#pmdarima.arima.ADFTest) |
| Stationarity | Kwiatkowski-Phillips-Schmidt-Shin | `from pmdarima.arima import KPSSTest` | [KPSS](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.KPSSTest.html)                      |
| Stationarity | Phillips-Peron                    | `from pmdarima.arima import PPTest`   | [PP](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html)                          |
| Seasonality  | Osborn-Chui-Smith-Birchenhall     | `from pmdarima.arima import OCSBTest` | [OCSB](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.html)                      |
| Seasonality  | Canova-Hansen                     | `from pmdarima.arima import CHTest`   | [CH](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html)                          |
| Correlation  | Auto-Correlation                  | `from pmdarima.utils import acf`      | [ACF](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.acf.html#pmdarima.utils.acf)         |
| Correlation  | Partial Auto-Ccorrelation         | `from pmdarima.utils import pacf`     | [PACF](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.pacf.html)                          |
-->

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
