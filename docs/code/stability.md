# Test the `stability` of a given Time-Series Dataset

## Introduction

!!! abstract "Summary"

    !!! quote "As stated by [Bocharov, Chickering & Heckerman](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/DATA06014FU1.pdf)"

        In technical terms the cases of long range forecasting instability are characterized by rapid growth of the mean absolute prediction error with time, which may or may not be accompanied by significant growth of the predicted standard deviation. In practice, the cases of instability where predicted standard deviation stays tame are especially misleading, since they can furnish unreliable predictions with little or no visual cues that would characterize them as unreliable.

        ---

        :material-arrow-right-bold: For more info, see: [Stability analysis of time series forecasting with ART models](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/DATA06014FU1.pdf).

    !!! info "Info"

        The test for stability is a measure of how much the data varies over each period of time. If a data-set is to be 'stable' that means that the means of each time period do not vary dramatically over time. In other words, the higher the variance between the means of each time-period, the more unstable the data is.

        There are two tests that can be used: The test for Stability and for Lumpiness (see sources [here](https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html#lumpiness_stability) and [here](https://www.rdocumentation.org/packages/tsfeatures/versions/1.0.1/topics/stl_features)). While the Stability test measures the variance of the means, the Lumpiness test measures the variance of the variances. For both of these measures, they simply indicate the extent to which each series varies by. The limits for this test are between $0$ and $1$; whereby a score of $0$ would indicate a perfectly stable (or perfectly smooth) data set, while a score of $1$ would indicate a completely unstable (or completely sporadic) data set.

        ---

        :material-arrow-right-bold: For more info, see: [The Future of Australian Energy Prices: Time-Series Analysis of Historic Prices and Forecast for Future Prices](https://chrimaho.medium.com/ausenergyprices-737b9cbe5540).

    !!! question "Source Library"

        The [`tsfeatures`](https://github.com/Nixtla/tsfeatures/blob/master/tsfeatures/tsfeatures.py) package was chosen because **REASONS**.

    !!! example "Source Module"

        All of the source code can be found within this modules:

        - [`ts_stat_tests.algorithms.stability`][ts_stat_tests.algorithms.stability].
        - [`ts_stat_tests.tests.stability`][ts_stat_tests.tests.stability].

## Stability Tests

::: ts_stat_tests.tests.stability
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source

## Stability Algorithms

::: ts_stat_tests.algorithms.stability
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
