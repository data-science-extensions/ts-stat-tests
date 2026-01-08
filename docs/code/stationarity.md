# Test the `stationarity` of a given Time-Series Dataset

## Introduction


!!! abstract "Summary"

    !!! quote "As stated by [Someone](#):"

        Stuff...

        ---

        :material-arrow-right-bold: For more info, see: [Source](#).

    !!! info "Info"

        There are actually three really good libraries which implements these tests:

        | library     | category     | algorithm                                     | short | import script                                      | url                                                                                                   |
        | ----------- | ------------ | --------------------------------------------- | ----- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
        | pmdarima    | Stationarity | Augmented Dickey-Fuller                       | ADF   | `from pmdarima.arima import ADFTest`               | https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ADFTest.html#pmdarima.arima.ADFTest |
        |             | Stationarity | Kwiatkowski-Phillips-Schmidt-Shin             | KPSS  | `from pmdarima.arima import KPSSTest`              | https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.KPSSTest.html                       |
        |             | Stationarity | Phillips-Peron                                | PP    | `from pmdarima.arima import PPTest`                | https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html                         |
        |             | Seasonality  | Osborn-Chui-Smith-Birchenhall                 | OCSB  | `from pmdarima.arima import OCSBTest`              | https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.html                       |
        |             | Seasonality  | Canova-Hansen                                 | CH    | `from pmdarima.arima import CHTest`                | https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html                         |
        |             | Correlation  | Auto-Correlation                              | ACF   | `from pmdarima.utils import ACFTest`               | https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.acf.html#pmdarima.utils.acf         |
        |             | Correlation  | Partial Auto-Ccorrelation                     | PACF  | `from pmdarima.utils import PACFTest`              | https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.pacf.html                           |
        | statsmodels | Stationarity | Augmented Dickey-Fuller                       | ADF   | `from statsmodels.api import adfuller`             | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html                  |
        |             | Stationarity | Kwiatkowski-Phillips-Schmidt-Shin             | KPSS  | `from statsmodels.api import kpss`                 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html                      |
        |             | Unit-Root    | Zivot-Andrews structural-break unit-root test | KPSS  | `from statsmodels.api import zivot_andrews`        | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html             |
        |             | Stationarity | Range unit-root test for stationarity         | RUR   | `from statsmodels.api import range_unit_root_test` | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html      |
        |             | White-Noise  | Ljung-Box Q Statistic                         | LB    | `from statsmodels.api import q_stat`               | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.q_stat.html                    |
        |             | Correlation  | Auto-Correlation                              | ACF   | `from statsmodels.api import acf`                  | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html                       |
        |             | Correlation  | Partial Auto-Ccorrelation                     | PACF  | `from statsmodels.api import pacf`                 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html                      |
        | arch        |

        ---

        :material-arrow-right-bold: For more info, see: [Source](#).

    !!! question "Source Library"

        The [`library`](#) package was chosen because **REASONS**.

    !!! example "Source Module"

        All of the source code can be found within this modules:

        - [`src.ts_stat_tests.algorithms.suitability`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/algorithms/suitability.py).
        - [`src.ts_stat_tests.tests.suitability`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/tests/suitability.py).

## Stationarity Tests

::: src.ts_stat_tests.tests.stationarity
    options:
        show_root_heading: false
        heading_level: 3
        show_if_no_docstrings: true
        member_order: source
        filters:
            - "^is"

## Stationarity Algorithms

::: src.ts_stat_tests.algorithms.stationarity
    options:
        show_root_heading: false
        heading_level: 3
        show_if_no_docstrings: true
        member_order: source
        filters:
            - "!^is"
            - "!^_"
