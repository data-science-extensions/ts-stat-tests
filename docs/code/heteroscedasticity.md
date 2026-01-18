# Test the `heteroscedasticity` of a given Time-Series Dataset

## Introduction

!!! abstract "Summary"

    !!! quote "From Wikipedia:"

        In statistics, a sequence of random variables is heteroscedastic if its random variables have different variances. The term means "differing variance" and comes from the Greek "hetero" (different) and "skedasis" (dispersion). In the context of regression analysis, heteroscedasticity refers to the case where the variability of a variable is unequal across the range of values of a second variable that predicts it.

        ---

        :material-arrow-right-bold: For more info, see: [Heteroscedasticity on Wikipedia](https://en.wikipedia.org/wiki/Heteroscedasticity).

    !!! info "Info"

        The library currently wraps four widely used heteroscedasticity tests from `statsmodels`, providing a unified interface for diagnostic checking:

        | library     | category           | algorithm            | short | import script                                                 | url                                                                                               |
        | ----------- | ------------------ | -------------------- | ----- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
        | statsmodels | Heteroscedasticity | ARCH Test            | ARCH  | `from statsmodels.stats.diagnostic import het_arch`           | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html           |
        |             | Heteroscedasticity | Breusch-Pagan Test   | BP    | `from statsmodels.stats.diagnostic import het_breuschpagan`   | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html   |
        |             | Heteroscedasticity | Goldfeld-Quandt Test | GQ    | `from statsmodels.stats.diagnostic import het_goldfeldquandt` | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_goldfeldquandt.html |
        |             | Heteroscedasticity | White's Test         | WHITE | `from statsmodels.stats.diagnostic import het_white`          | https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_white.html          |

        ---

        :material-arrow-right-bold: For more info, see: [Statsmodels Diagnostics](https://www.statsmodels.org/stable/stats.html#residual-diagnostics-and-specification-tests).

    !!! question "Source Library"

        `statsmodels` was chosen as the primary engine for these tests because it provides comprehensive, well-documented, and numerically stable implementations of the most common heteroscedasticity diagnostics. These tests allow for robust verification of OLS assumptions across various scenarios, including autoregressive clusters (ARCH) and general non-constant variance (White's).

    !!! example "Source Module"

        All of the source code can be found within these modules:

        - [`ts_stat_tests.algorithms.heteroscedasticity`][ts_stat_tests.algorithms.heteroscedasticity].
        - [`ts_stat_tests.tests.heteroscedasticity`][ts_stat_tests.tests.heteroscedasticity].

## Heteroscedasticity Tests

::: ts_stat_tests.tests.heteroscedasticity
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source

## Heteroscedasticity Algorithms

::: ts_stat_tests.algorithms.heteroscedasticity
    options:
        extra:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstrings: true
            member_order: source
