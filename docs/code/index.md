# Code Documentation

Welcome to the technical documentation for `ts-stat-tests`. This section provides detailed API references for all modules, information on our quality assurance processes, and live code coverage reports.

For a detailed analysis of the underlying mathematical models and a full catalog of supported tests, please see the [Algorithms Guide](../usage/algorithms.md).

## Key Features

- **Unified Interface**: Each statistical test is accessible through a standardised dispatcher function, providing a consistent user experience across different algorithms.
- **Boolean Checkers**: Each test module includes a corresponding `is_<test>()` function that returns a simple Boolean result, indicating whether the null hypothesis is accepted or rejected at a specified significance level.
- **Comprehensive Documentation**: Each module is thoroughly documented with theoretical backgrounds, algorithmic details, and practical usage examples, ensuring users can easily understand and apply the tests.
- **Robust Testing Suite**: The package includes an extensive suite of unit tests, achieving 100% code coverage to ensure reliability and correctness of all implementations.
- **Error Handling**: Standardised error handling across all modules ensures that users receive clear and informative feedback when inputs do not meet the required specifications.
- **Type Safety**: The use of strict type-checking with `@typechecked` decorators ensures that function inputs and outputs adhere to expected types, reducing runtime errors and improving code reliability.
- **Extensibility**: The modular design allows for easy addition of new statistical tests and algorithms in the future, facilitating ongoing development and enhancement of the library.
- **Integration with Established Libraries**: The package leverages well-known libraries such as `statsmodels`, `pmdarima`, `tsfeatures`, and `antropy` to implement the statistical tests, ensuring that users benefit from established and validated methodologies.
- **Quality Assurance**: All code passes stringent quality gates, including `Pylint` scores of 10/10 and zero `Pyright` errors, ensuring high standards of code quality and maintainability.


## Modules

| Module                                      | Description                                                                            |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| [Correlation](correlation.md)               | Tests to measure autocorrelation and cross-correlation in time series data.            |
| [Regularity](regularity.md)                 | Tests to measure the complexity and regularity of time series data.                    |
| [Seasonality](seasonality.md)               | Tests to identify seasonal patterns and measure seasonal strength in time series data. |
| [Stability](stability.md)                   | Tests to measure the stability and lumpiness of time series data.                      |
| [Stationarity](stationarity.md)             | Tests to determine if a time series is stationary or possesses unit roots.             |
| [Normality](normality.md)                   | Tests to assess if the residuals of a time series model follow a normal distribution.  |
| [Linearity](linearity.md)                   | Tests to evaluate linear specifications in time-series models.                         |
| [Heteroscedasticity](heteroscedasticity.md) | Tests to detect heteroscedasticity in the residuals of time series models.             |

## Testing

This package maintains 100% test coverage with comprehensive testing against:

1. **Code Style**: All code adheres to [`black`][black]formatting, including all code chunks in docstrings using the [`blacken-docs`][blacken-docs].
2. **Spell Checking**: All documentation and code comments are spell-checked using [`codespell`][codespell].
3. **Type Safety**: All code is type-checked using [`ty`][ty] and [`pyright`][pyright] and guarded during runtype by using the [`typeguard`][typeguard] library.
4. **Import Sorting**: All imports are sorted and managed using [`isort`][isort] and unused imports are removed using [`pycln`][pycln].
5. **Code Quality**: All code is checked for quality using [`pylint`][pylint] maintaining a score of 10/10, and checked for complexity using [`complexipy`][complexipy].
6. **Docstring Quality**: All docstrings are checked for style and completeness using [`docstring-format-checker`][docstring-format-checker].
7. **Unit Testing**: All code is unit-tested using [`pytest`][pytest] achieving 100% code coverage across the entire codebase, and including all examples in all docstrings tested using [`doctest`][doctest].
8. **Build Testing**: The package is built with [`uv`][uv] and the docs are built with [`mkdocs`][mkdocs] to ensure there are no build errors.


[black]: https://black.readthedocs.io
[blacken-docs]: https://github.com/adamchainz/blacken-docs
[ty]: https://docs.astral.sh/ty/
[pyright]: https://microsoft.github.io/pyright/
[typeguard]: https://typeguard.readthedocs.io
[isort]: https://pycqa.github.io/isort/
[pycln]: https://hadialqattan.github.io/pycln/
[codespell]: https://github.com/codespell-project/codespell
[pylint]: https://pylint.readthedocs.io/
[complexipy]: https://rohaquinlop.github.io/complexipy/
[docstring-format-checker]: https://data-science-extensions.com/toolboxes/docstring-format-checker/
[pytest]: https://docs.pytest.org/
[doctest]: https://docs.python.org/3/library/doctest.html
[uv]: https://docs.astral.sh/uv/
[mkdocs]: https://www.mkdocs.org/


## Coverage

<div style="position:relative; border:none; width:100%; height:100%; display:block; overflow:auto;">
    <iframe src="./coverage/index.html" style="width:100%; height:600px;"></iframe>
</div>


[statsmodels]: https://www.statsmodels.org
[pmdarima]: https://alkaline-ml.com/pmdarima
[arch]: https://arch.readthedocs.io
[tsfeatures]: https://github.com/Nixtla/tsfeatures
[antropy]: https://raphaelvallat.com/antropy
[scipy]: https://docs.scipy.org/
