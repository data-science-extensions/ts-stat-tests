# ============================================================================ #
#                                                                              #
#     Title: Regularity Algorithms                                             #
#     Purpose: Functions to compute regularity measures for time series data.  #
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
    This module contains algorithms to compute regularity measures for time series data, including approximate entropy, sample entropy, spectral entropy, and permutation entropy.
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
from typing import Literal, Optional, Union

# ## Python Third Party Imports ----
import numpy as np
from antropy import (
    app_entropy as a_app_entropy,
    perm_entropy as a_perm_entropy,
    sample_entropy as a_sample_entropy,
    spectral_entropy as a_spectral_entropy,
    svd_entropy as a_svd_entropy,
)
from numpy.typing import ArrayLike, NDArray
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = [
    "approx_entropy",
    "sample_entropy",
    "spectral_entropy",
    "permutation_entropy",
    "svd_entropy",
]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


VALID_KDTREE_METRIC_OPTIONS = Literal[
    "euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity"
]


VALID_SPECTRAL_ENTROPY_METHOD_OPTIONS = Literal["fft", "welch"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def approx_entropy(
    x: ArrayLike,
    order: int = 2,
    tolerance: Optional[float] = None,
    metric: VALID_KDTREE_METRIC_OPTIONS = "chebyshev",
) -> float:
    r"""
    !!! note "Summary"
        Approximate entropy is a measure of the amount of regularity or predictability in a time series. It is used to quantify the degree of self-similarity of a signal over different time scales, and can be useful for detecting underlying patterns or trends in data.

        This function implements the [`app_entropy()`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html) function from the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ abstract "Details"
        Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability of fluctuations over time-series data. Smaller values indicate that the data is more regular and predictable.

        To calculate approximate entropy, we first need to define a window size or scale factor, which determines the length of the subsequences that are used to compare the similarity of the time series. We then compare all possible pairs of subsequences within the time series and calculate the probability that two subsequences are within a certain tolerance level of each other, where the tolerance level is usually expressed as a percentage of the standard deviation of the time series.

        The approximate entropy is then defined as the negative natural logarithm of the average probability of similarity across all possible pairs of subsequences, normalized by the length of the time series and the scale factor.

        The approximate entropy measure is useful in a variety of applications, such as the analysis of physiological signals, financial time series, and climate data. It can be used to detect changes in the regularity or predictability of a time series over time, and can provide insights into the underlying dynamics or mechanisms that generate the signal.

    Params:
        x (ArrayLike):
            One-dimensional time series of shape `(n_times,)`.
        order (int, optional):
            Embedding dimension.<br>
            Defaults to `2`.
        tolerance (Optional[float], optional):
            Tolerance level or similarity criterion. If `None` (default), it is set to $0.2 \times \text{std}(x)$.<br>
            Defaults to `None`.
        metric (VALID_KDTREE_METRIC_OPTIONS, optional):
            Name of the distance metric function used with [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance). For a full list of all available metrics, see [`sklearn.metrics.pairwise.distance_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) and [`scipy.spatial.distance`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)<br>
            Defaults to `"chebyshev"`.

    Returns:
        (float):
            The approximate entropy score.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.regularity import approx_entropy
        >>> from ts_stat_tests.utils.data import data_airline, data_random
        >>> airline = data_airline.values
        >>> random = data_random

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Airline Passengers Data"}
        >>> print(f"{approx_entropy(x=airline):.4f}")
        0.6451

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Random Data"}
        >>> print(f"{approx_entropy(x=random):.4f}")
        1.8177

        ```

    ??? equation "Calculation"
        The equation for ApEn is:

        $$
        \text{ApEn}(m, r, N) = \phi_m(r) - \phi_{m+1}(r)
        $$

        where:

        - $m$ is the embedding dimension,
        - $r$ is the tolerance or similarity criterion,
        - $N$ is the length of the time series, and
        - $\phi_m(r)$ and $\phi_{m+1}(r)$ are the logarithms of the probabilities that two sequences of $m$ data points in the time series that are similar to each other within a tolerance $r$ remain similar for the next data point, for $m$ and $m+1$, respectively.

    ??? note "Notes"
        - **Inputs**: `x` is a 1-dimensional array. It represents time-series data, ideally with each element in the array being a measurement or value taken at regular time intervals.
        - **Settings**: `order` is used for determining the number of values that are used to construct each permutation pattern. If the embedding dimension is too small, we may miss important patterns. If it's too large, we may overfit noise.
        - **Metric**: The Chebyshev metric is often used because it is a robust and computationally efficient way to measure the distance between two time series.

    ??? success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ??? question "References"
        - [Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049](https://journals.physiology.org/doi/epdf/10.1152/ajpheart.2000.278.6.H2039)
        - [SK-Learn: Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics)
        - [Spatial data structures and algorithms](https://docs.scipy.org/doc/scipy/tutorial/spatial.html)

    ??? tip "See Also"
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
    """
    return a_app_entropy(
        x=x,
        order=order,
        tolerance=tolerance,
        metric=metric,
    )


@typechecked
def sample_entropy(
    x: ArrayLike,
    order: int = 2,
    tolerance: Optional[float] = None,
    metric: VALID_KDTREE_METRIC_OPTIONS = "chebyshev",
) -> float:
    r"""
    !!! note "Summary"
        Sample entropy is a measure of the amount of regularity or predictability in a time series. It is used to quantify the degree of self-similarity of a signal over different time scales, and can be useful for detecting underlying patterns or trends in data.

        This function implements the [`sample_entropy()`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html) function from the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ abstract "Details"
        Sample entropy is a modification of approximate entropy, used for assessing the complexity of physiological time-series signals. It has two advantages over approximate entropy: data length independence and a relatively trouble-free implementation. Large values indicate high complexity whereas smaller values characterize more self-similar and regular signals.

        The value of SampEn ranges from zero ($0$) to infinity ($\infty$), with lower values indicating higher regularity or predictability in the time series. A time series with high $SampEn$ is more unpredictable or irregular, whereas a time series with low $SampEn$ is more regular or predictable.

        Sample entropy is often used in time series forecasting to assess the complexity of the data and to determine whether a time series is suitable for modeling with a particular forecasting method, such as ARIMA or neural networks.

        Choosing an appropriate embedding dimension is crucial in ensuring that the permutation entropy calculation is robust and reliable, and captures the essential features of the time series in a meaningful way. This allows us to make more accurate and informative inferences about the behavior of the system that generated the data, and can be useful in a wide range of applications, from signal processing to data analysis and beyond.

    Params:
        x (ArrayLike):
            One-dimensional time series of shape `(n_times,)`.
        order (int, optional):
            Embedding dimension.<br>
            Defaults to `2`.
        tolerance (Optional[float], optional):
            Tolerance level or similarity criterion. If `None` (default), it is set to $0.2 \times \text{std}(x)$.<br>
            Defaults to `None`.
        metric (VALID_KDTREE_METRIC_OPTIONS, optional):
            Name of the distance metric function used with [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance). For a full list of all available metrics, see [`sklearn.metrics.pairwise.distance_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) and [`scipy.spatial.distance`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)<br>
            Defaults to `"chebyshev"`.

    Returns:
        (float):
            The sample entropy score.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.regularity import sample_entropy
        >>> from ts_stat_tests.utils.data import data_airline, data_random
        >>> airline = data_airline.values
        >>> random = data_random

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Airline Passengers Data"}
        >>> print(f"{sample_entropy(x=airline):.4f}")
        0.6177

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Random Data"}
        >>> print(f"{sample_entropy(x=random):.4f}")
        2.2017

        ```

    ??? equation "Calculation"
        The equation for sample entropy (SampEn) is as follows:

        $$
        \text{SampEn}(m, r, N) = - \log \left( \frac {C_m(r)} {C_{m+1}(r)} \right)
        $$

        where:

        - $m$ is the embedding dimension,
        - $r$ is the tolerance or similarity criterion,
        - $N$ is the length of the time series, and
        - $C_m(r)$ and $C_{m+1}(r)$ are the number of $m$-tuples (vectors of $m$ consecutive data points) that have a distance less than or equal to $r$, and $(m+1)$-tuples with the same property, respectively.

        The calculation of sample entropy involves the following steps:

        1. Choose the values of $m$ and $r$.
        2. Construct $m$-tuples from the time series data.
        3. Compute the number of $m$-tuples that are within a distance $r$ of each other ($C_m(r)$).
        4. Compute the number of $(m+1)$-tuples that are within a distance $r$ of each other ($C_{m+1}(r)$).
        5. Compute the value of $SampEn$ using the formula above.

    ??? note "Notes"
        - Note that if `metric == 'chebyshev'` and `len(x) < 5000` points, then the sample entropy is computed using a fast custom Numba script. For other distance metric or longer time-series, the sample entropy is computed using a code from the [`mne-features`](https://mne.tools/mne-features/) package by Jean-Baptiste Schiratti and Alexandre Gramfort (requires sklearn).
        - The embedding dimension is important in the calculation of sample entropy because it affects the sensitivity of the measure to different patterns in the data. If the embedding dimension is too small, we may miss important patterns or variations. If it is too large, we may overfit the data.

    ??? success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ??? question "References"
        - [Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049](https://journals.physiology.org/doi/epdf/10.1152/ajpheart.2000.278.6.H2039)
        - [SK-Learn: Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics)
        - [Spatial data structures and algorithms](https://docs.scipy.org/doc/scipy/tutorial/spatial.html)

    ??? tip "See Also"
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
        - [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)
        - [`sklearn.metrics.pairwise_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)
        - [`scipy.spatial.distance`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
    """
    return a_sample_entropy(
        x=x,
        order=order,
        tolerance=tolerance,
        metric=metric,
    )


@typechecked
def permutation_entropy(
    x: ArrayLike,
    order: int = 3,
    delay: Union[int, list, NDArray[np.int64]] = 1,
    normalize: bool = False,
) -> float:
    r"""
    !!! note "Summary"
        Permutation entropy is a measure of the complexity or randomness of a time series. It is based on the idea of permuting the order of the values in the time series and calculating the entropy of the resulting permutation patterns.

        This function implements the [`perm_entropy()`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html) function from the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ abstract "Details"
        The permutation entropy is a complexity measure for time-series first introduced by Bandt and Pompe in 2002.

        It is particularly useful for detecting nonlinear dynamics and nonstationarity in the data. The value of permutation entropy ranges from $0$ to $\log_2(\text{order}!)$, where the lower bound is attained for an increasing or decreasing sequence of values, and the upper bound for a completely random system where all possible permutations appear with the same probability.

        Choosing an appropriate embedding dimension is crucial in ensuring that the permutation entropy calculation is robust and reliable, and captures the essential features of the time series in a meaningful way.

    Params:
        x (ArrayLike):
            One-dimensional time series of shape `(n_times,)`.
        order (int, optional):
            Order of permutation entropy.<br>
            Defaults to `3`.
        delay (Union[int, list, NDArray[np.int64]], optional):
            Time delay (lag). If multiple values are passed, the average permutation entropy across all these delays is calculated.<br>
            Defaults to `1`.
        normalize (bool, optional):
            If `True`, divide by $\log_2(\text{order}!)$ to normalize the entropy between $0$ and $1$. Otherwise, return the permutation entropy in bits.<br>
            Defaults to `False`.

    Returns:
        (Union[float, NDArray[np.float64]]):
            The permutation entropy of the data set.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.regularity import permutation_entropy
        >>> from ts_stat_tests.utils.data import data_airline, data_random
        >>> airline = data_airline.values
        >>> random = data_random

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Airline Passengers Data"}
        >>> print(f"{permutation_entropy(x=airline):.4f}")
        2.3601

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Random Data (Normalized)"}
        >>> print(f"{permutation_entropy(x=random, normalize=True):.4f}")
        0.9997

        ```

    ??? equation "Calculation"
        The formula for permutation entropy ($PE$) is as follows:

        $$
        PE(n) = - \sum_{i=0}^{n!} p(i) \times \log_2(p(i))
        $$

        where:

        - $n$ is the embedding dimension (`order`),
        - $p(i)$ is the probability of the $i$-th ordinal pattern.

        The embedded matrix $Y$ is created by:

        $$
        \begin{align}
            y(i) &= [x_i, x_{i+\text{delay}}, \dots, x_{i+(\text{order}-1) \times \text{delay}}] \\
            Y &= [y(1), y(2), \dots, y(N-(\text{order}-1) \times \text{delay})]^T
        \end{align}
        $$

    ??? note "Notes"
        - The embedding dimension (`order`) determines the number of values used to construct each permutation pattern. If too small, patterns may be missed. If too large, overfitting to noise may occur.

    ??? success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ??? question "References"
        - [Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural complexity measure for time series." Physical review letters 88.17 (2002): 174102](http://materias.df.uba.ar/dnla2019c1/files/2019/03/permutation_entropy.pdf)

    ??? tip "See Also"
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
    """
    return a_perm_entropy(
        x=x,
        order=order,
        delay=delay,  # type: ignore[arg-type]  # antropy function can handle Union[int, list[int], NDArray[np.int64]], however the function signature is not annotated as such
        normalize=normalize,
    )


@typechecked
def spectral_entropy(
    x: ArrayLike,
    sf: float = 1,
    method: VALID_SPECTRAL_ENTROPY_METHOD_OPTIONS = "fft",
    nperseg: Optional[int] = None,
    normalize: bool = False,
    axis: int = -1,
) -> Union[float, NDArray[np.float64]]:
    r"""
    !!! note "Summary"
        Spectral entropy is a measure of the amount of complexity or unpredictability in a signal's frequency domain representation. It is used to quantify the degree of randomness or regularity in the power spectrum of a signal.

        This function implements the [`spectral_entropy()`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html) function from the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ abstract "Details"
        Spectral Entropy is defined to be the Shannon entropy of the power spectral density (PSD) of the data. It is based on the Shannon entropy, which is a measure of the uncertainty or information content of a probability distribution.

        The value of spectral entropy ranges from $0$ to $\log_2(N)$, where $N$ is the number of frequency bands. Lower values indicate a more concentrated or regular distribution of power, while higher values indicate a more spread-out or irregular distribution.

        Spectral entropy is particularly useful for detecting periodicity and cyclical patterns, as well as changes in the frequency distribution over time.

    Params:
        x (ArrayLike):
            One-dimensional or N-dimensional data array.
        sf (float, optional):
            Sampling frequency, in Hz.<br>
            Defaults to `1`.
        method (VALID_SPECTRAL_ENTROPY_METHOD_OPTIONS, optional):
            Spectral estimation method: `'fft'` or `'welch'`.<br>
            - `'fft'`: Fourier Transformation ([`scipy.signal.periodogram()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html#scipy.signal.periodogram))<br>
            - `'welch'`: Welch periodogram ([`scipy.signal.welch()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch))<br>
            Defaults to `"fft"`.
        nperseg (Optional[int], optional):
            Length of each FFT segment for Welch method. If `None`, uses `scipy`'s default of 256 samples.<br>
            Defaults to `None`.
        normalize (bool, optional):
            If `True`, divide by $\log_2(\text{psd.size})$ to normalize the spectral entropy to be between $0$ and $1$. Otherwise, return the spectral entropy in bits.<br>
            Defaults to `False`.
        axis (int, optional):
            The axis along which the entropy is calculated. Default is the last axis.<br>
            Defaults to `-1`.

    Returns:
        (Union[float, NDArray[np.float64]]):
            The spectral entropy score. Returned as a float for 1D input, or a numpy array for N-dimensional input.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.regularity import spectral_entropy
        >>> from ts_stat_tests.utils.data import data_airline
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Airline Passengers Data"}
        >>> print(f"{spectral_entropy(x=airline, sf=12):.4f}")
        2.6538

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Welch method for spectral entropy"}
        >>> data_sine = np.sin(2 * np.pi * 1 * np.arange(400) / 100)
        >>> print(f"{spectral_entropy(x=data_sine, sf=100, method='welch'):.4f}")
        1.2938

        ```

    ??? equation "Calculation"
        The spectral entropy ($SE$) is defined as:

        $$
        H(x, f_s) = - \sum_{i=0}^{f_s/2} P(i) \times \log_2(P(i))
        $$

        where:

        - $P(i)$ is the normalized power spectral density (PSD) at the $i$-th frequency band,
        - $f_s$ is the sampling frequency.

    ??? note "Notes"
        - The power spectrum represents the energy of the signal at different frequencies. High spectral entropy indicates multiple sources or processes with different frequencies, while low spectral entropy suggests a dominant frequency or periodicity.

    ??? success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ??? question "References"
        - [Inouye, T. et al. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum. Electroencephalography and clinical neurophysiology, 79(3), 204-210.](https://pubmed.ncbi.nlm.nih.gov/1714811/)
        - [Wikipedia: Spectral density](https://en.wikipedia.org/wiki/Spectral_density)
        - [Wikipedia: Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method)

    ??? tip "See Also"
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
    """
    return a_spectral_entropy(
        x=x,
        sf=sf,
        method=method,
        nperseg=nperseg,
        normalize=normalize,
        axis=axis,
    )


@typechecked
def svd_entropy(
    x: ArrayLike,
    order: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> float:
    r"""
    !!! note "Summary"
        SVD entropy is a measure of the complexity or randomness of a time series based on Singular Value Decomposition (SVD).

        This function implements the [`svd_entropy()`](https://raphaelvallat.com/antropy/build/html/generated/antropy.svd_entropy.html) function from the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ abstract "Details"
        SVD entropy is calculated by first embedding the time series into a matrix, then performing SVD on that matrix to obtain the singular values. The entropy is then calculated from the normalized singular values.

    Params:
        x (ArrayLike):
            One-dimensional time series of shape `(n_times,)`.
        order (int, optional):
            Order of the SVD entropy (embedding dimension).<br>
            Defaults to `3`.
        delay (int, optional):
            Time delay (lag).<br>
            Defaults to `1`.
        normalize (bool, optional):
            If `True`, divide by $\log_2(\text{order}!)$ to normalize the entropy between $0$ and $1$.<br>
            Defaults to `False`.

    Returns:
        (float):
            The SVD entropy of the data set.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.regularity import svd_entropy
        >>> from ts_stat_tests.utils.data import data_random
        >>> random = data_random

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic SVD entropy"}
        >>> print(f"{svd_entropy(random):.4f}")
        1.3514

        ```

    ??? equation "Calculation"
        The SVD entropy is calculated as the Shannon entropy of the singular values of the embedded matrix.

    ??? note "Notes"
        - Singular Value Decomposition (SVD) is a factorization of a real or complex matrix. It is the generalization of the eigendecomposition of a positive semidefinite normal matrix.

    ??? success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ??? tip "See Also"
        - [`antropy.svd_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.svd_entropy.html)
        - [`ts_stat_tests.algorithms.regularity.approx_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.regularity.sample_entropy`][ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`ts_stat_tests.algorithms.regularity.permutation_entropy`][ts_stat_tests.algorithms.regularity.permutation_entropy]
        - [`ts_stat_tests.algorithms.regularity.spectral_entropy`][ts_stat_tests.algorithms.regularity.spectral_entropy]
    """
    return a_svd_entropy(
        x=x,
        order=order,
        delay=delay,
        normalize=normalize,
    )
