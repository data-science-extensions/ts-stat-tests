# Test the `correlation` of a given Time-Series Dataset

## Introduction

!!! abstract "Summary"

    !!! quote "As stated by [Anais Dotis-Georgiou](https://www.influxdata.com/blog/author/anais):"

        The term autocorrelation refers to the degree of similarity between A) a given time series, and B) a lagged version of itself, over C) successive time intervals. In other words, autocorrelation is intended to measure the relationship between a variable's present value and any past values that you may have access to.

        Therefore, a time series autocorrelation attempts to measure the current values of a variable against the historical data of that variable. It ultimately plots one series over the other, and determines the degree of similarity between the two.

        For the sake of comparison, autocorrelation is essentially the exact same process that you would go through when calculating the correlation between two different sets of time series values on your own. The major difference here is that autocorrelation uses the same time series two times: once in its original values, and then again once a few different time periods have occurred.

        Autocorrelation is also known as serial correlation, time series correlation and lagged correlation. Regardless of how it's being used, autocorrelation is an ideal method for uncovering trends and patterns in time series data that would have otherwise gone undiscovered.

        ---

        :material-arrow-right-bold: For more info, see: [InfluxData: Autocorrelation in Time Series Data](https://www.influxdata.com/blog/autocorrelation-in-time-series-data/).

    !!! info "Info"

        An important test to do on Time-Series data is to measure it's level of Auto-Correlation (McMurry & Politis, 2010; Hyndman, nd.(b)). While 'correlation' refers to how two variables change based on the other's value, 'auto-correlation' is how a variable changes based on it's own value over time (the phrase "auto" refers to "self"). For the Auto-Correlation Function, it uses a '`lag`' function. For example, a lag value of `0` is 100% correlated, which is logical, because that is it's own value; whereas a lag value of `1` or greater, the level of auto-correlation decreases as it gets further away from `lag0`.

        For well-structured time-series data sets, it would be expected to see a conical-shaped Auto-Correlation plot. If it were not a well-structured time-series data set, then this Auto-Correlation plot would look more like white noise, and there would not be any logical shape. The blue dotted lines are included as a reference point for determining if any of the observations are significantly different from zero.

        Moreover, analysis of the data's Auto-Correlation (ACF) should be combined with analysis of its Partial Auto-Correlation (PACF). While the ACF is the "direct" relationship between an observation and it's relevant lag observation, the PACF removes the "indirect" relationship between these observations. Effectively, the Partial Auto-Correlation between `lag1` and `lag5` is the "actual" correlation between these two observations, after removing the influence that `lag2`, `lag3`, and `lag4` has on `lag5`.

        What this means is that the Partial Auto-Correlation plot would have a very high value at `lag0`, which will drop very quickly at `lag1`, and should remain below the blue reference lines for the remainder of the Correlogram. The observations of `lag>0` should resemble white noise data points. If it does not resemble white noise, and there is a distinct pattern occurring, then the data is not suitable for time-series forecasting.

        | library     | category    | algorithm                        | short | import script                                                    | url                                                                                                  |
        | ----------- | ----------- | -------------------------------- | ----- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
        | statsmodels | Correlation | Autocorrelation Function         | ACF   | `from statsmodels.tsa.stattools import acf`                      | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html                      |
        |             | Correlation | Partial Autocorrelation Function | PACF  | `from statsmodels.tsa.stattools import pacf`                     | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html                     |
        |             | Correlation | Cross-Correlation Function       | CCF   | `from statsmodels.tsa.stattools import ccf`                      | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html                      |
        |             | Correlation | Ljung-Box Test                   | LB    | `from statsmodels.stats.diagnostic import acorr_ljungbox`        | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html        |
        |             | Correlation | Lagrange Multiplier Test         | LM    | `from statsmodels.stats.diagnostic import acorr_lm`              | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html              |
        |             | Correlation | Breusch-Godfrey LM Test          | BGLM  | `from statsmodels.stats.diagnostic import acorr_breusch_godfrey` | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html |

        ---

        :material-arrow-right-bold: For more info, see: [Statsmodels Diagnostic](https://www.statsmodels.org/stable/stats.html#diagnostic-tests).

    !!! question "Source Library"

        The `statsmodels` package was chosen because it provides mature, well-tested implementations of core time-series tools (such as ACF, PACF, and correlograms), integrates seamlessly with `numpy` and `pandas` data structures, and offers a comprehensive suite of statistical tests that align closely with the methods demonstrated in this project.

    !!! example "Source Module"

        All of the source code can be found within these modules:

        - [`ts_stat_tests.algorithms.correlation`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/algorithms/correlation.py).
        - [`ts_stat_tests.tests.correlation`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/tests/correlation.py).

## Correlation Tests

::: ts_stat_tests.tests.correlation
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source

## Correlation Algorithms

::: ts_stat_tests.algorithms.correlation
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
