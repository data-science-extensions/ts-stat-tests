# Test the `normality` of a given Time-Series Dataset

## Introduction

!!! abstract "Summary"

    !!! quote "As stated by the [NIST/SEMATECH e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35.htm):"

        Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known. Their importance is partly due to the central limit theorem.

        ---

        :material-arrow-right-bold: For more info, see: [Engineering Statistics Handbook: Measures of Skewness and Kurtosis](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm).

    !!! info "Info"

        The normality test is used to determine whether a data set is well-modeled by a normal distribution. In time series forecasting, we primarily test the residuals (errors) of a model for normality. If the residuals follow a normal distribution, it suggests that the model has successfully captured the systematic patterns in the data, and the remaining errors are random white noise.

        If the residuals are not normally distributed, it may indicate that the model is missing important features, such as seasonal patterns or long-term trends, or that a transformation of the data (e.g., Log or Box-Cox) is required before modeling.

        | library     | category  | algorithm                   | short | import script                                            | url                                                                                          |
        | ----------- | --------- | --------------------------- | ----- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
        | scipy       | Normality | Shapiro-Wilk Test           | SW    | `from scipy.stats import shapiro`                        | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html                |
        | scipy       | Normality | D'Agostino & Pearson's Test | DP    | `from scipy.stats import normaltest`                     | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html             |
        | scipy       | Normality | Anderson-Darling Test       | AD    | `from scipy.stats import anderson`                       | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html               |
        | statsmodels | Normality | Jarque-Bera Test            | JB    | `from statsmodels.stats.stattools import jarque_bera`    | https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.jarque_bera.html    |
        | statsmodels | Normality | Omnibus Test                | OB    | `from statsmodels.stats.diagnostic import omni_normtest` | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.omni_normtest.html |

        ---

        :material-arrow-right-bold: For more info, see: [Hyndman & Athanasopoulos: Forecasting: Principles and Practice](https://otexts.com/fpp3/residuals.html).

    !!! question "Source Library"

        The [`scipy`](https://scipy.org/) and [`statsmodels`](https://www.statsmodels.org/) packages were chosen because they provide standard, reliable implementations of classical statistical tests. `scipy.stats` provides implementations for Shapiro-Wilk, D'Agostino-Pearson, and Anderson-Darling tests, while `statsmodels` provides the Jarque-Bera and Omnibus tests.

    !!! example "Source Module"

        All of the source code can be found within these modules:

        - [`ts_stat_tests.normality.algorithms`][ts_stat_tests.normality.algorithms].
        - [`ts_stat_tests.normality.tests`][ts_stat_tests.normality.tests].



## Modules


::: ts_stat_tests.normality.tests
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source


::: ts_stat_tests.normality.algorithms
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
