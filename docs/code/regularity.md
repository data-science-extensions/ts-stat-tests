# Test the `regularity` of a given Time-Series Dataset

## Introduction

!!! abstract "Summary"

    !!! quote "As stated by [Selva Prabhakaran](https://www.machinelearningplus.com/author/selva86/):"

        The more regular and repeatable patterns a time series has, the easier it is to forecast.

        The 'Approximate Entropy' algorithm can be used to quantify the regularity and unpredictability of fluctuations in a time series.

        The higher the approximate entropy, the more difficult it is to forecast it.

        Another better alternate is the 'Sample Entropy'.

        Sample Entropy is similar to approximate entropy but is more consistent in estimating the complexity even for smaller time series.

        For example, a random time series with fewer data points can have a lower 'approximate entropy' than a more 'regular' time series, whereas, a longer random time series will have a higher 'approximate entropy'.

        ---

        :material-arrow-right-bold: For more info, see: [Time Series Analysis in Python: A Comprehensive Guide with Examples](https://www.machinelearningplus.com/time-series/time-series-analysis-python/).

    !!! info "Info"

        To state that the data is 'regular' is to say that the data points are evenly spaced, regularly collected, and not missing data points (ie. do not contain excessive `NA` values). Logically, it is not always necessary to conduct the Test for Regularity on automatically collected data (like for example with Energy Prices, or Daily Temperature), however if this data was collected manually then it is highly recommended. If the data does not meet the requirements of Regularity, then it is necessary to return to the data collection plan, and revise the methodology used.

        | library | category   | algorithm           | short  | import script                          | url                                                                                  |
        | ------- | ---------- | ------------------- | ------ | -------------------------------------- | ------------------------------------------------------------------------------------ |
        | antropy | Regularity | Approximate Entropy | AppEn  | `from antropy import app_entropy`      | https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html      |
        | antropy | Regularity | Sample Entropy      | SampEn | `from antropy import sample_entropy`   | https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html   |
        | antropy | Regularity | Permutation Entropy | PermEn | `from antropy import perm_entropy`     | https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html     |
        | antropy | Regularity | Spectral Entropy    | SpecEn | `from antropy import spectral_entropy` | https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html |
        | antropy | Regularity | SVD Entropy         | SvdEn  | `from antropy import svd_entropy`      | https://raphaelvallat.com/antropy/build/html/generated/antropy.svd_entropy.html      |

        ---

        :material-arrow-right-bold: For more info, see: [The Future of Australian Energy Prices: Time-Series Analysis of Historic Prices and Forecast for Future Prices](https://chrimaho.medium.com/ausenergyprices-737b9cbe5540).

    !!! question "Source Library"

        The [`AntroPy`](https://raphaelvallat.com/antropy/build/html/index.html) package was chosen because it provides well-tested and efficient implementations of approximate entropy, sample entropy, and related complexity measures for time-series data, is built on top of the scientific Python stack (NumPy/SciPy), and is actively maintained and open source, making it a reliable choice for reproducible statistical analysis.

    !!! example "Source Module"

        All of the source code can be found within the modules:

        - [`ts_stat_tests.algorithms.regularity`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/algorithms/regularity.py).
        - [`ts_stat_tests.tests.regularity`](https://github.com/chrimaho/ts-stat-tests/blob/main/src/ts_stat_tests/tests/regularity.py).

## Regularity Tests

::: ts_stat_tests.tests.regularity
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
            filters:
                - "^is"

## Regularity Algorithms

::: ts_stat_tests.tests.regularity
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
            filters:
                - "!^is"
                - "!^_"

::: ts_stat_tests.algorithms.regularity
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
            filters:
                - "_entropy"
