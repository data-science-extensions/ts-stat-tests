# Test the `stationarity` of a given Time-Series Dataset

## Introduction


!!! abstract "Summary"

    !!! quote "As stated by [Robert Nau, Duke University](https://people.duke.edu/~rnau/411arim.htm):"

        A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time. Most statistical forecasting methods are based on the assumption that the time series can be rendered approximately stationary (i.e., "stationarized") through the use of mathematical transformations.

        ---

        :material-arrow-right-bold: For more info, see: [Introduction to ARIMA models](https://people.duke.edu/~rnau/411arim.htm).

    !!! info "Info"

        There are two primary libraries used to implement these tests, ensuring both breadth of coverage and numerical reliability:

        | library     | category     | algorithm                                     | short | import script                                                | url                                                                                              |
        | ----------- | ------------ | --------------------------------------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
        | statsmodels | Stationarity | Augmented Dickey-Fuller                       | ADF   | `from statsmodels.tsa.stattools import adfuller`             | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html             |
        |             | Stationarity | Kwiatkowski-Phillips-Schmidt-Shin             | KPSS  | `from statsmodels.tsa.stattools import kpss`                 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html                 |
        |             | Unit-Root    | Zivot-Andrews structural-break unit-root test | ZA    | `from statsmodels.tsa.stattools import zivot_andrews`        | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html        |
        |             | Stationarity | Range unit-root test for stationarity         | RUR   | `from statsmodels.tsa.stattools import range_unit_root_test` | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html |
        | arch        | Stationarity | Phillips-Perron                               | PP    | `from arch.unitroot import PhillipsPerron`                   | https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.PhillipsPerron.html       |
        |             | Stationarity | Elliott-Rothenberg-Stock (ERS) de-trended DF  | ERS   | `from arch.unitroot import DFGLS`                            | https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html                |
        |             | Unit-Root    | Variance Ratio (VR) test                      | VR    | `from arch.unitroot import VarianceRatio`                    | https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html        |

        ---

        :material-arrow-right-bold: For more info, see: [Statsmodels TSA](https://www.statsmodels.org/stable/tsa.html) and [Arch Unit Roots](https://arch.readthedocs.io/en/latest/unitroot/unitroot.html).

    !!! question "Source Library"

        The `statsmodels` and `arch` packages were chosen because they provide robust, industry-standard implementations of unit root and stationarity tests. Specifically, `statsmodels` offers the classic ADF and KPSS tests along with specialized ones like Zivot-Andrews and RUR, while `arch` provides high-performance implementations of Phillips-Perron, ERS, and Variance Ratio tests, ensuring a comprehensive suite of tools for detecting non-stationarity and random walks.

    !!! example "Source Module"

        All of the source code can be found within these modules:

        - [`ts_stat_tests.algorithms.stationarity`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/algorithms/stationarity.py).
        - [`ts_stat_tests.tests.stationarity`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/tests/stationarity.py).

## Stationarity Tests

::: ts_stat_tests.tests.stationarity
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
            filters:
                - "^is"
                - "^stationarity"

## Stationarity Algorithms

::: ts_stat_tests.algorithms.stationarity
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
            filters:
                - "!^is"
                - "!^_"
