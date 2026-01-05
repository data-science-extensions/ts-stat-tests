# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Union

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.regularity import (
    approx_entropy,
    sample_entropy,
    spectral_entropy,
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
    metric: str = "chebyshev",
    sf: float = 1,
    normalize: bool = True,
) -> float:
    """
    !!! Summary "Summary"
        Test for the entropy of a given data set.

    ???+ Info "Details"
        This function is a convenience wrapper around the three underlying algorithms:<br>
        - [`approx_entropy()`][src.ts_stat_tests.algorithms.regularity.approx_entropy]<br>
        - [`sample_entropy()`][src.ts_stat_tests.algorithms.regularity.sample_entropy]<br>
        - [`spectral_entropy()`][src.ts_stat_tests.algorithms.regularity.spectral_entropy]

    Params:
        x (array_like):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str, optional):
            Which entropy algorithm to use.<br>
            - `sample_entropy()`: `["sample", "sampl", "samp"]`<br>
            - `approx_entropy()`: `["app", "aprox", "approx"]`<br>
            - `spectral_entropy()`: `["spec", "spect", "spectral"]`<br>
            Defaults to `"sample"`.
        order (int, optional):
            Embedding dimension.<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `2`.
        metric (str, optional):
            Name of the distance metric function used with [`sklearn.neighors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance).<br>
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
        - [`regularity()`][src.ts_stat_tests.tests.regularity.regularity]
        - [`approx_entropy()`][src.ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`sample_entropy()`][src.ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`spectral_entropy()`][src.ts_stat_tests.algorithms.regularity.spectral_entropy]
    """
    options: dict[str, tuple[str, ...]] = {
        "sampl": ("sample", "sampl", "samp"),
        "aprox": ("app", "aprox", "approx"),
        "spect": ("spec", "spect", "spectral"),
    }
    if algorithm in options["sampl"]:
        return sample_entropy(x=x, order=order, metric=metric)
    elif algorithm in options["aprox"]:
        return approx_entropy(x=x, order=order, metric=metric)
    elif algorithm in options["spect"]:
        return spectral_entropy(x=x, sf=sf, normalize=normalize)
    else:
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
    metric: str = "chebyshev",
    sf: float = 1,
    normalize: bool = True,
) -> float:
    """
    !!! Summary "Summary"
        Test for the regularity of a given data set.

    ???+ Info "Details"
        This is a pass-through, convenience wrapper around the [`entropy()`][src.ts_stat_tests.tests.regularity.entropy] function.

    Params:
        x (array_like):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str, optional):
            Which entropy algorithm to use.<br>
            - `sample_entropy()`: `["sample", "sampl", "samp"]`<br>
            - `approx_entropy()`: `["app", "aprox", "approx"]`<br>
            - `spectral_entropy()`: `["spec", "spect", "spectral"]`<br>
            Defaults to `"sample"`.
        order (int, optional):
            Embedding dimension.<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `2`.
        metric (str, optional):
            Name of the distance metric function used with [`sklearn.neighors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance).<br>
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

        `approx_entropy`:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> approx_entropy(x=data)
        0.6451264780416452
        ```

        ---

        `sample_entropy`:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> sample_entropy(x=data)
        0.6177074729583698
        ```

        ---

        `spectral_entropy`:
        ```pycon {.py .python linenums="1"  title="Basic usage"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> spectral_entropy(x=data, sf=1)
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
        - [`entropy()`][src.ts_stat_tests.tests.regularity.entropy]
        - [`approx_entropy()`][src.ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`sample_entropy()`][src.ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`spectral_entropy()`][src.ts_stat_tests.algorithms.regularity.spectral_entropy]
    """
    return entropy(x=x, algorithm=algorithm, order=order, metric=metric, sf=sf, normalize=normalize)


@typechecked
def is_regular(
    x: ArrayLike,
    algorithm: str = "sample",
    order: int = 2,
    sf: float = 1,
    metric: str = "chebyshev",
    normalize: bool = True,
    tolerance: Union[str, float, int, None] = "default",
) -> dict[str, Union[str, float, bool]]:
    """
    !!! Summary "Summary"
        Test whether a given data set is `regular` or not.

    ???+ Info "Details"
        This function implements the given algorithm (defined in the parameter `algorithm`), and returns a dictionary containing the relevant data:
        ```python
        {
            "result": ...,  # The result of the test. Will be `True` if `entropy<tolerance`, and `False` otherwise
            "entropy": ...,  # A `float` value, the result of the `entropy()` function
            "tolerance": ...,  # A `float` value, which is the tolerance used for determining whether or not the `entropy` is `regular` or not
        }
        ```

    Params:
        x (array_like):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str, optional):
            Which entropy algorithm to use.<br>
            - `sample_entropy()`: `["sample", "sampl", "samp"]`<br>
            - `approx_entropy()`: `["app", "aprox", "approx"]`<br>
            - `spectral_entropy()`: `["spec", "spect", "spectral"]`<br>
            Defaults to `"sample"`.
        order (int, optional):
            Embedding dimension.<br>
            Only relevant when `algorithm=sample` or `algorithm=approx`.<br>
            Defaults to `2`.
        metric (str, optional):
            Name of the distance metric function used with [`sklearn.neighors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance).<br>
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
            - If `tolerance` is either `"default"` or `None`, then `tolerance` will be derrived from `x` using the calculation:
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
        >>> sample_entropy(x=data, algorithm="sample")
        {"entropy": 0.6177074729583698, "tolerance": 23.909808306554297, "result": True}
        ```

        ```pycon {.py .python linenums="1" title="Approx Entropy"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> approx_entropy(x=data, algorithm="approx", tolerance=20)
        {"entropy": 0.6451264780416452, "tolerance": 20, "result": True}
        ```

        ```pycon {.py .python linenums="1"  title="Spectral Entropy"}
        >>> from sktime.datasets import load_airline
        >>> data = load_airline()
        >>> spectral_entropy(x=data, algorithm="spectral", sf=1)
        {"entropy": 0.4287365561752448, "tolerance": 23.909808306554297, "result": True}
        ```

    ??? Question "References"
        - Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
        - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        - Inouye, T. et al. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum. Electroencephalography and clinical neurophysiology, 79(3), 204-210.
        - https://en.wikipedia.org/wiki/Spectral_density
        - https://en.wikipedia.org/wiki/Welch%27s_method

    ??? Tip "See Also"
        - [`entropy()`][src.ts_stat_tests.tests.regularity.entropy]
        - [`regularity()`][src.ts_stat_tests.tests.regularity.regularity]
        - [`approx_entropy()`][src.ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`sample_entropy()`][src.ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`spectral_entropy()`][src.ts_stat_tests.algorithms.regularity.spectral_entropy]
    """
    if isinstance(tolerance, (float, int)):
        tol = tolerance
    elif tolerance in ["default", None]:
        tol = 0.2 * np.std(a=x)
    else:
        raise ValueError(
            f"Invalid option for `tolerance` parameter: {tolerance}.\n"
            f"Valid options are:\n"
            f"- A number with type `float` or `int`,\n"
            f"- A string with value `default`,\n"
            f"- The value `None`."
        )
    value = regularity(x=x, order=order, sf=sf, metric=metric, algorithm=algorithm, normalize=normalize)
    result = True if value < tol else False
    return {"result": result, "entropy": value, "tolerance": tol}
