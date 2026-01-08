# ============================================================================ #
#                                                                              #
#     Title: Regularity Tests                                                  #
#     Purpose: Convenience functions for regularity algorithms.                #
#                                                                              #
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Overview                                                              ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#  Description                                                              ####
# ---------------------------------------------------------------------------- #


"""
!!! note "Summary"
    This module contains convenience functions and tests for regularity measures, allowing for easy access to different entropy algorithms.
"""


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Union, cast

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.regularity import (
    VALID_KDTREE_METRIC_OPTIONS,
    approx_entropy,
    permutation_entropy,
    sample_entropy,
    spectral_entropy,
    svd_entropy,
)
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["entropy", "regularity", "is_regular"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def entropy(
    x: ArrayLike,
    algorithm: str = "sample",
    order: int = 2,
    metric: VALID_KDTREE_METRIC_OPTIONS = "chebyshev",
    sf: float = 1,
    normalize: bool = True,
) -> float:
    """
    !!! note "Summary"
        Test for the entropy of a given data set.

    ???+ abstract "Details"
        This function is a convenience wrapper around the five underlying algorithms:<br>
        - [`approx_entropy()`][ts_stat_tests.algorithms.regularity.approx_entropy]<br>
        - [`sample_entropy()`][ts_stat_tests.algorithms.regularity.sample_entropy]<br>
        - [`spectral_entropy()`][ts_stat_tests.algorithms.regularity.spectral_entropy]<br>
        - [`permutation_entropy()`][ts_stat_tests.algorithms.regularity.permutation_entropy]<br>
        - [`svd_entropy()`][ts_stat_tests.algorithms.regularity.svd_entropy]

    Params:
        x (ArrayLike):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str, optional):
            Which entropy algorithm to use.<br>
            - `sample_entropy()`: `["sample", "sampl", "samp"]`<br>
            - `approx_entropy()`: `["app", "approx"]`<br>
            - `spectral_entropy()`: `["spec", "spect", "spectral"]`<br>
            - `permutation_entropy()`: `["perm", "permutation"]`<br>
            - `svd_entropy()`: `["svd", "svd_entropy"]`<br>
            Defaults to `"sample"`.
        order (int, optional):
            Embedding dimension.<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `2`.
        metric (VALID_KDTREE_METRIC_OPTIONS):
            Name of the distance metric function used with [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance).<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `"chebyshev"`.
        sf (float, optional):
            Sampling frequency, in Hz.<br>
            Only relevant when `algorithm=spectral`.<br>
            Defaults to `1`.
        normalize (bool, optional):
            If `True`, divide by $log2(psd.size)$ to normalize the spectral entropy to be between $0$ and $1$. Otherwise, return the spectral entropy in bit.<br>
            Only relevant when `algorithm=spectral`.<br>
            Defaults to `True`.

    Raises:
        ValueError: When the given value for `algorithm` is not valid.

    Returns:
        (float):
            The Entropy value.

    !!! Success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ Example "Examples"

        `approx_entropy`:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> entropy(x=data, algorithm="approx")
        0.6451264780416452
        ```

        ---

        `sample_entropy`:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> entropy(x=data, algorithm="sample")
        0.6177074729583698
        ```

        ---

        `spectral_entropy`:
        ```pycon {.py .python linenums="1"  title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> entropy(x=data, algorithm="spectral", sf=1)
        2.6538040647031726
        ```

        ```pycon {.py .python linenums="1"  title="Advanced usage"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> spectral_entropy(data, 2, "welch", normalize=True)
        0.3371369604224553
        ```

    ??? Question "References"
        - Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
        - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        - Inouye, T. et al. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum. Electroencephalography and clinical neurophysiology, 79(3), 204-210.
        - https://en.wikipedia.org/wiki/Spectral_density
        - https://en.wikipedia.org/wiki/Welch%27s_method

    ??? Tip "See Also"
        - [`regularity()`][ts_stat_tests.tests.regularity.regularity]
        - [`approx_entropy()`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`sample_entropy()`][ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`spectral_entropy()`][ts_stat_tests.algorithms.regularity.spectral_entropy]
        - [`permutation_entropy()`][ts_stat_tests.algorithms.regularity.permutation_entropy]
        - [`svd_entropy()`][ts_stat_tests.algorithms.regularity.svd_entropy]
    """
    options: dict[str, tuple[str, ...]] = {
        "sampl": ("sample", "sampl", "samp"),
        "approx": ("app", "approx"),
        "spect": ("spec", "spect", "spectral"),
        "perm": ("perm", "permutation"),
        "svd": ("svd", "svd_entropy"),
    }
    if algorithm in options["sampl"]:
        return sample_entropy(x=x, order=order, metric=metric)
    if algorithm in options["approx"]:
        return approx_entropy(x=x, order=order, metric=metric)
    if algorithm in options["spect"]:
        return cast(float, spectral_entropy(x=x, sf=sf, normalize=normalize))
    if algorithm in options["perm"]:
        return permutation_entropy(x=x, order=order, normalize=normalize)
    if algorithm in options["svd"]:
        return svd_entropy(x=x, order=order, normalize=normalize)
    raise ValueError(
        generate_error_message(
            parameter_name="algorithm",
            value_parsed=algorithm,
            options=options,
        )
    )


@typechecked
def regularity(
    x: ArrayLike,
    algorithm: str = "sample",
    order: int = 2,
    metric: VALID_KDTREE_METRIC_OPTIONS = "chebyshev",
    sf: float = 1,
    normalize: bool = True,
) -> float:
    """
    !!! note "Summary"
        Test for the regularity of a given data set.

    ???+ abstract "Details"
        This is a pass-through, convenience wrapper around the [`entropy()`][ts_stat_tests.tests.regularity.entropy] function.

    Params:
        x (ArrayLike):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str, optional):
            Which entropy algorithm to use.<br>
            - `sample_entropy()`: `["sample", "sampl", "samp"]`<br>
            - `approx_entropy()`: `["app", "approx"]`<br>
            - `spectral_entropy()`: `["spec", "spect", "spectral"]`<br>
            - `permutation_entropy()`: `["perm", "permutation"]`<br>
            - `svd_entropy()`: `["svd", "svd_entropy"]`<br>
            Defaults to `"sample"`.
        order (int, optional):
            Embedding dimension.<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `2`.
        metric (VALID_KDTREE_METRIC_OPTIONS):
            Name of the distance metric function used with [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance).<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `"chebyshev"`.
        sf (float, optional):
            Sampling frequency, in Hz.<br>
            Only relevant when `algorithm=spectral`.<br>
            Defaults to `1`.
        normalize (bool, optional):
            If `True`, divide by $log2(psd.size)$ to normalize the spectral entropy to be between $0$ and $1$. Otherwise, return the spectral entropy in bit.<br>
            Only relevant when `algorithm=spectral`.<br>
            Defaults to `True`.

    Returns:
        (float):
            The Regularity value.

    !!! Success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ Example "Examples"

        `regularity` with `approx_entropy` algorithm:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> from ts_stat_tests.tests.regularity import regularity
        >>> data = load_airline()
        >>> regularity(x=data, algorithm="approx_entropy")
        0.6451264780416452
        ```

        ---

        `regularity` with `sample_entropy` algorithm:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> from ts_stat_tests.tests.regularity import regularity
        >>> data = load_airline()
        >>> regularity(x=data, algorithm="sample_entropy")
        0.6177074729583698
        ```

        ---

        `regularity` with `spectral_entropy` algorithm:
        ```pycon {.py .python linenums="1"  title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> from ts_stat_tests.tests.regularity import regularity
        >>> data = load_airline()
        >>> regularity(x=data, algorithm="spectral_entropy", sf=1)
        2.6538040647031726
        ```

        ```pycon {.py .python linenums="1"  title="Advanced usage"}
        >>> from sktime.datasets import load_airline
        >>> from ts_stat_tests.tests.regularity import regularity
        >>> data = load_airline()
        >>> regularity(data, algorithm="spectral_entropy", sf=2, method="welch", normalize=True)
        0.3371369604224553
        ```

    ??? Question "References"
        - Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
        - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        - Inouye, T. et al. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum. Electroencephalography and clinical neurophysiology, 79(3), 204-210.
        - https://en.wikipedia.org/wiki/Spectral_density
        - https://en.wikipedia.org/wiki/Welch%27s_method

    ??? Tip "See Also"
        - [`entropy()`][ts_stat_tests.tests.regularity.entropy]
        - [`approx_entropy()`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`sample_entropy()`][ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`spectral_entropy()`][ts_stat_tests.algorithms.regularity.spectral_entropy]
        - [`permutation_entropy()`][ts_stat_tests.algorithms.regularity.permutation_entropy]
        - [`svd_entropy()`][ts_stat_tests.algorithms.regularity.svd_entropy]
    """
    return entropy(x=x, algorithm=algorithm, order=order, metric=metric, sf=sf, normalize=normalize)


@typechecked
def is_regular(
    x: ArrayLike,
    algorithm: str = "sample",
    order: int = 2,
    sf: float = 1,
    metric: VALID_KDTREE_METRIC_OPTIONS = "chebyshev",
    normalize: bool = True,
    tolerance: Union[str, float, int, None] = "default",
) -> dict[str, Union[str, float, bool]]:
    """
    !!! note "Summary"
        Test whether a given data set is `regular` or not.

    ???+ abstract "Details"
        This function implements the given algorithm (defined in the parameter `algorithm`), and returns a dictionary containing the relevant data:
        ```python
        {
            "result": ...,  # The result of the test. Will be `True` if `entropy<tolerance`, and `False` otherwise
            "entropy": ...,  # A `float` value, the result of the `entropy()` function
            "tolerance": ...,  # A `float` value, which is the tolerance used for determining whether or not the `entropy` is `regular` or not
        }
        ```

    Params:
        x (ArrayLike):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str, optional):
            Which entropy algorithm to use.<br>
            - `sample_entropy()`: `["sample", "sampl", "samp"]`<br>
            - `approx_entropy()`: `["app", "approx"]`<br>
            - `spectral_entropy()`: `["spec", "spect", "spectral"]`<br>
            - `permutation_entropy()`: `["perm", "permutation"]`<br>
            - `svd_entropy()`: `["svd", "svd_entropy"]`<br>
            Defaults to `"sample"`.
        order (int, optional):
            Embedding dimension.<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `2`.
        metric (VALID_KDTREE_METRIC_OPTIONS):
            Name of the distance metric function used with [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance).<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `"chebyshev"`.
        sf (float, optional):
            Sampling frequency, in Hz.<br>
            Only relevant when `algorithm=spectral`.<br>
            Defaults to `1`.
        normalize (bool, optional):
            If `True`, divide by $log2(psd.size)$ to normalize the spectral entropy to be between $0$ and $1$. Otherwise, return the spectral entropy in bit.<br>
            Only relevant when `algorithm=spectral`.<br>
            Defaults to `True`.
        tolerance (Union[str, float, int, None], optional):
            The tolerance value used to determine whether or not the result is `regular` or not.<br>
            - If `tolerance` is either type `int` or `float`, then this value will be used.<br>
            - If `tolerance` is either `"default"` or `None`, then `tolerance` will be derived from `x` using the calculation:
                ```python
                tolerance = 0.2 * np.std(a=x)
                ```
            - If any other value is given, then a `ValueError` error will be raised.<br>
            Defaults to `"default"`.

    Raises:
        (ValueError): If the given `tolerance` parameter is invalid.

            Valid options are:

            - A number with type `float` or `int`, or
            - A string with value `default`, or
            - The value `None`.

    Returns:
        (Dict[str, Union[str, float, bool]]):
            A dictionary with only 3 keys containing the results of the test:
            ```python
            {
                "result": ...,
                "entropy": ...,
                "tolerance": ...,
            }
            ```

    !!! Success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ Example "Examples"

        ```pycon {.py .python linenums="1" title="Sample Entropy"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> is_regular(x=data, algorithm="sample")
        {"entropy": 0.6177074729583698, "tolerance": 23.909808306554297, "result": True}
        ```

        ```pycon {.py .python linenums="1" title="Approx Entropy"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> is_regular(x=data, algorithm="approx", tolerance=20)
        {"entropy": 0.6451264780416452, "tolerance": 20, "result": True}
        ```

        ```pycon {.py .python linenums="1"  title="Spectral Entropy"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> is_regular(x=data, algorithm="spectral", sf=1)
        {"entropy": 0.4287365561752448, "tolerance": 23.909808306554297, "result": True}
        ```

    ??? Question "References"
        - Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
        - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        - Inouye, T. et al. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum. Electroencephalography and clinical neurophysiology, 79(3), 204-210.
        - https://en.wikipedia.org/wiki/Spectral_density
        - https://en.wikipedia.org/wiki/Welch%27s_method

    ??? Tip "See Also"
        - [`entropy()`][ts_stat_tests.tests.regularity.entropy]
        - [`regularity()`][ts_stat_tests.tests.regularity.regularity]
        - [`approx_entropy()`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`sample_entropy()`][ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`spectral_entropy()`][ts_stat_tests.algorithms.regularity.spectral_entropy]
        - [`permutation_entropy()`][ts_stat_tests.algorithms.regularity.permutation_entropy]
        - [`svd_entropy()`][ts_stat_tests.algorithms.regularity.svd_entropy]
    """
    if isinstance(tolerance, (float, int)):
        tol = tolerance
    elif tolerance in ["default", None]:
        tol = 0.2 * np.std(a=np.asarray(x))
    else:
        raise ValueError(
            f"Invalid option for `tolerance` parameter: {tolerance}.\n"
            f"Valid options are:\n"
            f"- A number with type `float` or `int`,\n"
            f"- A string with value `default`,\n"
            f"- The value `None`."
        )
    value = regularity(x=x, order=order, sf=sf, metric=metric, algorithm=algorithm, normalize=normalize)
    result = value < tol
    return {
        "result": bool(result),
        "entropy": float(value),
        "tolerance": float(tol),
    }
