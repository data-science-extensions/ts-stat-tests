# Test the `seasonality` of a given Time-Series Dataset


## Introduction

!!! abstract "Summary"

    !!! quote "As stated by [Rob Hyndman and George Athanasopoulos](https://otexts.com/fpp2/)"

        In describing these time series, we have used words such as "trend" and "seasonal" which need to be defined more carefully.

        **Trend**

        - A _trend_ exists when there is a long-term increase or decrease in the data. It does not have to be linear. Sometimes we will refer to a trend as "changing direction", when it might go from an increasing trend to a decreasing trend.

        **Seasonal**

        - A _seasonal_ pattern occurs when a time series is affected by seasonal factors such as the time of the year or the day of the week. Seasonality is always of a fixed and known frequency. The monthly sales of antidiabetic drugs above shows seasonality which is induced partly by the change in the cost of the drugs at the end of the calendar year.

        **Cyclic**

        - A _cycle_ occurs when the data exhibit rises and falls that are not of a fixed frequency. These fluctuations are usually due to economic conditions, and are often related to the "business cycle". The duration of these fluctuations is usually at least 2 years.

        Many people confuse cyclic behaviour with seasonal behaviour, but they are really quite different. If the fluctuations are not of a fixed frequency then they are cyclic; if the frequency is unchanging and associated with some aspect of the calendar, then the pattern is seasonal. In general, the average length of cycles is longer than the length of a seasonal pattern, and the magnitudes of cycles tend to be more variable than the magnitudes of seasonal patterns.

        Many time series include trend, cycles and seasonality. When choosing a forecasting method, we will first need to identify the time series patterns in the data, and then choose a method that is able to capture the patterns properly.

        ---

        :material-arrow-right-bold: For more info, see: [Forecasting: Principles and Practice - Time Series Patterns](https://otexts.com/fpp2/tspatterns.html)

    !!! info "Info"

        For a really good article on CH & OCSB tests, check: [When A Time Series Only Quacks Like A Duck: Testing for Stationarity Before Running Forecast Models. With Python. And A Duckling Picture.](https://towardsdatascience.com/when-a-time-series-only-quacks-like-a-duck-10de9e165e)

        By declaring that the data is 'seasonal' is to say that the peaks and troughs are the same for each of the periods, and that the future seasonal periods could be predicted with a reasonable level of confidence.

    !!! question "Source Library"

        The [`pmdarima`](https://alkaline-ml.com/pmdarima) package was chosen for the `ch` and `ocsb` tests, while the [`tsfeatures`](https://github.com/Nixtla/tsfeatures/blob/master/tsfeatures/tsfeatures.py) package was chosen for the `seasonal_strength`, `trend_strength` and `spikiness` tests. Because **REASONS**. The [`seastests`](https://rdrr.io/cran/seastests) package (from the R language) was chosen as inspiration for the `qs` test.

    !!! example "Source Module"

        All of the source code can be found within the modules:

        - [`src.ts_stat_tests.algorithms.seasonality`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/algorithms/seasonality.py).
        - [`src.ts_stat_tests.tests.seasonality`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/tests/seasonality.py).


## Seasonality Tests

::: ts_stat_tests.tests.seasonality
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
            filters:
                - "^is"
                - "^seasonality"


## Seasonality Algorithms

::: ts_stat_tests.algorithms.seasonality
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
            filters:
                - "!^_"
