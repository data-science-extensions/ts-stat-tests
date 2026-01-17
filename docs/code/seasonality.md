# Test the `seasonality` of a given Time-Series Dataset


## Introduction

!!! abstract "Summary"

    !!! quote "As stated by [Rob Hyndman and George Athanasopoulos](https://otexts.com/fpp2/tspatterns.html)"

        In describing these time series, we have used words such as "trend" and "seasonal" which need to be defined more carefully.

        **Trend**
        A _trend_ exists when there is a long-term increase or decrease in the data. It does not have to be linear. Sometimes we will refer to a trend as "changing direction", when it might go from an increasing trend to a decreasing trend.

        **Seasonal**
        A _seasonal_ pattern occurs when a time series is affected by seasonal factors such as the time of the year or the day of the week. Seasonality is always of a fixed and known frequency.

        **Cyclic**
        A _cycle_ occurs when the data exhibit rises and falls that are not of a fixed frequency. These fluctuations are usually due to economic conditions, and are often related to the "business cycle". The duration of these fluctuations is usually at least 2 years.

        ---

        :material-arrow-right-bold: For more info, see: [Forecasting: Principles and Practice - Time Series Patterns](https://otexts.com/fpp2/tspatterns.html)

    | Information        | Details                                                                |
    | :----------------- | :--------------------------------------------------------------------- |
    | **Module**         | [ts_stat_tests.tests.seasonality][ts_stat_tests.tests.seasonality]     |
    | **Algorithms**     | `qs`, `ocsb`, `ch`, `seasonal_strength`, `trend_strength`, `spikiness` |
    | **Complexity**     | $O(n \log n)$ to $O(n^2)$ depending on algorithm                       |
    | **Implementation** | `statsmodels`, `pmdarima`, `tsfeatures`                                |

    !!! info "Source Library"

        We leverage several industry-standard libraries for the underlying statistical tests in this module.

        1.  **[`pmdarima`](https://alkaline-ml.com/pmdarima)**: Provides the implementation for the Canova-Hansen (`ch`) and Osborn-Chui-Smith-Birchenhall (`ocsb`) tests through its `nsdiffs` and estimator classes.
        2.  **[`tsfeatures`](https://github.com/Nixtla/tsfeatures)**: Used as the basis for calculating seasonal strength, trend strength, and spikiness, following the approach popularized by the R `feasts` and `tsfeatures` packages.
        3.  **[`seastests`](https://rdrr.io/cran/seastests)**: An R package that served as the primary reference and inspiration for our Python implementation of the Quenouille-Sarle (`qs`) test.

    !!! info "Source Module"

        The source code for the seasonality tests is organized into two primary layers:

        - **[`src.ts_stat_tests.algorithms.seasonality`][ts_stat_tests.algorithms.seasonality]**: Contains the core mathematical implementations and wrappers for third-party libraries.
        - **[`src.ts_stat_tests.tests.seasonality`][ts_stat_tests.tests.seasonality]**: Provides the top-level user interface, including the `seasonality` dispatcher and the `is_seasonal` boolean check.


## Seasonality Tests

::: ts_stat_tests.tests.seasonality
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source


## Seasonality Algorithms

::: ts_stat_tests.algorithms.seasonality
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
