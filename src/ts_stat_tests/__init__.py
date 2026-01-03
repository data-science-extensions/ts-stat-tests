"""
Time Series Statistical Tests

A collection of statistical tests for time series data.
"""

# ## Python StdLib Imports ----
from importlib.metadata import PackageMetadata, metadata


_metadata: PackageMetadata = metadata("ts-stat-tests")
__name__: str = _metadata["Name"]
__version__: str = _metadata["Version"]
__author__: str = _metadata["Author"]
__email__: str = _metadata["Author-email"]
