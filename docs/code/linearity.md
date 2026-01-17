# Test the `linearity` of a given Time-Series Dataset

## Introduction

!!! abstract "Summary"

    Linearity testing evaluates whether the relationship between variables in a model follows a linear specification. In the context of time-series analysis, these tests help determine if linear regression models are appropriate or if non-linear dynamics (such as structural breaks or exponential trends) are present.

    !!! quote "As stated by [James Ramsey (1969)](https://doi.org/10.1111/j.2517-6161.1969.tb00796.x):"

        Specification errors, such as the omission of relevant variables, incorrect functional form, or correlation between the regressor and the error term, can lead to biased and inconsistent estimators. The RESET test provides a robust method for detecting these departures from linearity.

        ---

        :material-arrow-right-bold: For more info, see: [Ramsey RESET test](https://en.wikipedia.org/wiki/Ramsey_RESET_test).

    !!! info "Implementation Details"

        The suite of tests provided here includes measures based on recursive residuals, subset fitting, and power transformations. These tests are typically applied to the residuals of an estimated OLS model to verify that no significant non-linear information remains in the error term.

        ---

        :material-arrow-right-bold: For more info, see: [Statsmodels Linearity Tests](https://www.statsmodels.org/stable/stats.html#diagnostic-tests).

    !!! question "Source Library"

        The [`statsmodels`](https://www.statsmodels.org/stable/index.html) package was chosen because it provides the industry-standard implementation of econometric diagnostic tests, ensuring mathematical rigor and consistency with well-established statistical literature.

    !!! example "Source Module"

        All of the source code can be found within these modules:

        - [`ts_stat_tests.algorithms.linearity`][ts_stat_tests.algorithms.linearity].
        - [`ts_stat_tests.tests.linearity`][ts_stat_tests.tests.linearity].

## Linearity Tests

::: ts_stat_tests.tests.linearity
    options:
        extras:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstring: true
            members_order: source

## Linearity Algorithms

::: ts_stat_tests.algorithms.linearity
    options:
        extras:
            show_root_heading: false
            heading_level: 3
            show_if_no_docstring: true
            members_order: source
