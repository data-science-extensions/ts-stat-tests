# ============================================================================ #
#                                                                              #
#     Title: Regularity                                                        #
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
from typing import Any, Literal, Optional, Union, cast

# ## Python Third Party Imports ----
import numpy as np
from antropy import (
    app_entropy as a_app_entropy,
    perm_entropy as a_perm_entropy,
    sample_entropy as a_sample_entropy,
    spectral_entropy as a_spectral_entropy,
    svd_entropy as a_svd_entropy,
)
from numpy.typing import ArrayLike
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
# from sklearn.neighbors import KDTree; print(KDTree.valid_metrics);


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
    """
    !!! note "Summary"
        Approximate entropy is a measure of the amount of regularity or predictability in a time series. It is used to quantify the degree of self-similarity of a signal over different time scales, and can be useful for detecting underlying patterns or trends in data

    ???+ abstract "Details"
        Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability of fluctuations over time-series data. Smaller values indicates that the data is more regular and predictable.

        The tolerance value ($r$) is set to $0.2 \\times std(x)$.

        To calculate approximate entropy, we first need to define a window size or scale factor, which determines the length of the subsequences that are used to compare the similarity of the time series. We then compare all possible pairs of subsequences within the time series and calculate the probability that two subsequences are within a certain tolerance level of each other, where the tolerance level is usually expressed as a percentage of the standard deviation of the time series.

        The approximate entropy is then defined as the negative natural logarithm of the average probability of similarity across all possible pairs of subsequences, normalized by the length of the time series and the scale factor.

        The approximate entropy measure is useful in a variety of applications, such as the analysis of physiological signals, financial time series, and climate data. It can be used to detect changes in the regularity or predictability of a time series over time, and can provide insights into the underlying dynamics or mechanisms that generate the signal. For example, a decrease in approximate entropy may indicate the onset of a disease or a shift in the underlying physiological state, while an increase in approximate entropy may suggest the presence of noise or other external influences on the system.

        The equation for ApEn is:

        $$
        ApEn(m, r, N) = φm(r) - φm+1(r)
        $$

        where:

        - $m$ is the embedding dimension,
        - $r$ is the tolerance or similarity criterion,
        - $N$ is the length of the time series, and
        - $φm(r)$ and $φm+1(r)$ are the logarithms of the probabilities that two sequences of m data points in the time series that are similar to each other within a tolerance $r$ remain similar for the next data point, for $m$ and $m+1$, respectively.

        ```
        ApEn(m, r, N) = φm(r) - φm+1(r)
        ```

        The calculation of ApEn involves the following steps:

        1. Create a set of vectors, each containing $m$ data points from the time series, where $m$ is the embedding dimension.
        1. Calculate the Euclidean distance between each pair of vectors and count the number of pairs that are within a distance $r$ of each other.
        1. Compute the probabilities $φm(r)$ and $φm+1(r)$ using the counts from step 2.
        1. Compute $ApEn(m, r, N)$ using the equation above.

        The value of ApEn ranges from zero ($0$) to infinity ($\\infty$), with lower values indicating higher regularity or predictability in the time series. A time series with high ApEn is more unpredictable or irregular, whereas a time series with low ApEn is more regular or predictable.

        ApEn is often used in time series forecasting to assess the complexity of the data and to determine whether a time series is suitable for modeling with a particular forecasting method, such as ARIMA or neural networks.

        When calculating the Approximate entropy requires the specification of a set of parameters that determine the characteristics of the time series. One of these parameters is the _embedding dimension_, which determines the number of values that are used to construct each permutation pattern.

        The embedding dimension is important in the calculation of permutation entropy because it affects the sensitivity of the measure to different patterns in the data. If the embedding dimension is too small, we may miss important patterns or variations in the time series, and the resulting permutation entropy value may not accurately reflect the underlying complexity of the signal. On the other hand, if the embedding dimension is too large, we may overfit the data and produce a permutation entropy value that is overly sensitive to noise or other random fluctuations.

        Choosing an appropriate embedding dimension is therefore crucial in ensuring that the permutation entropy calculation is robust and reliable, and captures the essential features of the time series in a meaningful way. This allows us to make more accurate and informative inferences about the behavior of the system that generated the data, and can be useful in a wide range of applications, from signal processing to data analysis and beyond.

    Params:
        x (ArrayLike):
            One-dimensional time series of shape (n_times).
        order (int, optional):
            Embedding dimension.<br>
            Defaults to `2`.
        tolerance (Optional[float]):
            Tolerance level or similarity criterion. If `None` (default), it is set to $0.2 \times std(x)$.<br>
            Defaults to `None`.
        metric (VALID_KDTREE_METRIC_OPTIONS):
            Name of the distance metric function used with [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance). For a full list of all available metrics, see [`sklearn.metrics.pairwise.distance_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) and [`scipy.spatial.distance`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)<br>
            Defaults to `"chebyshev"`.

    Returns:
        (float):
            Approximate Entropy score.

    ???+ note "Notes"

        **Inputs**:

        - `x` is a 1-dimensional array.<br>
            It represents time-series data, ideally with each element in the array being a measurement or value taken at regular time intervuls over the length of the array. The exact number of elements (or length of the array) can realistically be greater than 30 elements long, but ideally should be a few hundred elements long, and is even better when it is a few thousand elements long.

        **Settings**:

        - `order` is used for determining the number of values that are used to construct each permutation pattern.
            - If the embedding dimension is too small, we may miss important patterns or variations in the time series, and the resulting approximate entropy value may not accurately reflect the underlying complexity of the signal.
            - If the embedding dimension is too large, we may overfit the data and produce an approximate entropy value that is overly sensitive to noise or other random fluctuations.

        - `metric` is used for determining which distance metric to use for the underlying distance-space between two time series.
            - The Chebyshev metric is often used when calculating approximate entropy because it is a robust and computationally efficient way to measure the distance between two time series. The Chebyshev distance between two vectors is defined as the maximum absolute difference between their corresponding components. When comparing two subsequences in the time series, the Chebyshev distance is calculated by taking the maximum absolute difference between their corresponding values at each point in time.
            - The use of the Chebyshev metric in approximate entropy calculation has been found to be effective in detecting the presence of patterns or regularities in a time series. This is because the Chebyshev metric is less sensitive to outliers and noise in the data than other metrics, such as Euclidean distance or Manhattan distance.
            - However, other metrics can also be used to calculate approximate entropy, depending on the specific characteristics of the time series being analyzed and the research question at hand. For example, the Euclidean distance can be used as an alternative to the Chebyshev metric, especially when the time series is relatively smooth and does not contain sharp spikes or discontinuities.
            - In addition, other metrics such as Hamming distance or Cosine distance can be used in cases where the time series represents binary or categorical data, or when the time series has a natural geometric interpretation, respectively.
            - Ultimately, the choice of metric depends on the specific application and the properties of the time series being analyzed. The use of different metrics can lead to different results and insights, and it is important to carefully consider the advantages and limitations of each approach before making a decision.

        **Outputs**:

        - A single number is returned, which represents the entropy score. It will be a float value, where numbers close to $0$ indicate _less_ entropy; meaning that it is _more_ stable, regular and predictable.

        **Expectations**:

        - A returned value close to $0$ means that it is quite stable.
        - 'close to zero' is highly interpretable and context specific. For example, if you generate a sequence of random numbers between $0$ and $1$, then 'close to zero' would need to be down to 5 or 6 decimal places. But if you're looking at a sequence of numbers which range between $0$ and $1000$, then 'close to zero' would be anything less than, say, $20$.
        - When using other available metrics (such as `'euclidean'`, '`hamming'`, `'cosine'`, etc), the results may be a little unexpected. So, it's best to do some research around which metric is best to use for the specific purposes at hand.

    !!! Success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ Example "Examples"

        ```pycon {.py .python linenums="1" title="Prepare data"}
        >>> import numpy as np
        >>> from sktime.datasets import load_airline
        >>> from stochastic.processes import noise as sn
        >>> data_airline = load_airline()
        >>> rng = np.random.default_rng(seed=42)
        >>> data_noise = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
        >>> data_random = rng.random(1000)
        ```

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> print(f"{approx_entropy(x=data_airline):.4f}")
        0.6451
        ```

        ```pycon {.py .python linenums="1" title="Gaussian noise"}
        >>> print(f"{approx_entropy(x=data_noise, order=2):.4f}")
        2.1958
        ```

        ```pycon {.py .python linenums="1" title="Euclidean metric"}
        >>> print(f"{approx_entropy(x=data_noise, order=3, metric='euclidean'):.4f}")
        1.5120
        ```

        ```pycon {.py .python linenums="1" title="Random data"}
        >>> print(f"{approx_entropy(x=data_random):.4f}")
        1.8030
        ```

    ??? Question "References"
        - [Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049](https://journals.physiology.org/doi/epdf/10.1152/ajpheart.2000.278.6.H2039)
        - [SK-Learn: Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics)
        - [Spatial data structures and algorithms](https://docs.scipy.org/doc/scipy/tutorial/spatial.html)

    ??? Tip "See Also"
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
        - [`ts_stat_tests.algorithms.app_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.approx_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.perm_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.spectral_entropy`][ts_stat_tests.algorithms.regularity.spectral_entropy]
        - [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)
        - [`sklearn.metrics.pairwise_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)
        - [`scipy.spatial.distance`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
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
    """
    !!! note "Summary"
        Sample entropy is a measure of the amount of regularity or predictability in a time series. It is used to quantify the degree of self-similarity of a signal over different time scales, and can be useful for detecting underlying patterns or trends in data.

    ???+ abstract "Details"
        Sample entropy is a modification of approximate entropy, used for assessing the complexity of physiological time-series signals. It has two advantages over approximate entropy: data length independence and a relatively trouble-free implementation. Large values indicate high complexity whereas smaller values characterize more self-similar and regular signals.

        The equation for sample entropy (SampEn) is as follows:

        $$
        SampEn(m, r, N) = - \\log \\left( \\frac {Cm(r)} {Cm+1(r)} \\right)
        $$

        where:

        - $m$ is the embedding dimension,
        - $r$ is the tolerance or similarity criterion,
        - $N$ is the length of the time series, and
        - $Cm(r)$ and $Cm+1(r)$ are the number of $m$-tuples (vectors of $m$ consecutive data points) that have a distance less than or equal to $r$, and $(m+1)$-tuples with the same property, respectively. The log function is the natural logarithm.

        ```
        SampEn(m, r, N) = -log( Cm(r) / Cm+1(r) )
        ```

        The calculation of sample entropy involves the following steps:

        1. Choose the values of $m$ and $r$.
        1. Construct $m$-tuples from the time series data.
        1. Compute the number of $m$-tuples that are within a distance r of each other ($Cm(r)$).
        1. Compute the number of $(m+1)$-tuples that are within a distance r of each other ($Cm+1(r)$).
        1. Compute the value of $SampEn$ using the formula above.

        The value of SampEn ranges from zero ($0$) to infinity ($\\infty$), with lower values indicating higher regularity or predictability in the time series. A time series with high $SampEn$ is more unpredictable or irregular, whereas a time series with low $SampEn$ is more regular or predictable.

        Sample entropy is often used in time series forecasting to assess the complexity of the data and to determine whether a time series is suitable for modeling with a particular forecasting method, such as ARIMA or neural networks.

        Note that if `metric == 'chebyshev'` and `len(x) < 5000` points, then the sample entropy is computed using a fast custom Numba script. For other distance metric or longer time-series, the sample entropy is computed using a code from the [`mne-features`](https://mne.tools/mne-features/) package by Jean-Baptiste Schiratti and Alexandre Gramfort (requires sklearn).

        To calculate the Sample entropy requires the specification of a set of parameters that determine the characteristics of the time series. One of these parameters is the _embedding dimension_, which determines the number of values that are used to construct each permutation pattern.

        The embedding dimension is important in the calculation of permutation entropy because it affects the sensitivity of the measure to different patterns in the data. If the embedding dimension is too small, we may miss important patterns or variations in the time series, and the resulting permutation entropy value may not accurately reflect the underlying complexity of the signal. On the other hand, if the embedding dimension is too large, we may overfit the data and produce a permutation entropy value that is overly sensitive to noise or other random fluctuations.

        Choosing an appropriate embedding dimension is therefore crucial in ensuring that the permutation entropy calculation is robust and reliable, and captures the essential features of the time series in a meaningful way. This allows us to make more accurate and informative inferences about the behavior of the system that generated the data, and can be useful in a wide range of applications, from signal processing to data analysis and beyond.

    Params:
        x (ArrayLike):
            One-dimensional time series of shape (n_times).
        order (int, optional):
            Embedding dimension.<br>
            Defaults to `2`.
        tolerance (Optional[float]):
            Tolerance level or similarity criterion. If `None` (default), it is set to $0.2 \times std(x)$.<br>
            Defaults to `None`.
        metric (VALID_KDTREE_METRIC_OPTIONS):
            Name of the distance metric function used with [`sklearn.neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree). Default is to use the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance). For a full list of all available metrics, see [`sklearn.metrics.pairwise.distance_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) and [`scipy.spatial.distance`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)<br>
            Defaults to `"chebyshev"`.

    Returns:
        (float):
            Sample Entropy score.

    !!! Success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ Example "Examples"

        ```pycon {.py .python linenums="1" title="Prepare data"}
        >>> import numpy as np
        >>> from sktime.datasets import load_airline
        >>> from stochastic.processes import noise as sn
        >>> data_airline = load_airline()
        >>> rng = np.random.default_rng(seed=42)
        >>> data_noise = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
        >>> data_random = rng.random(1000)
        >>> data_sine = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
        >>> data_line = np.arange(1000)
        ```

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> print(f"{sample_entropy(x=data_airline):.4f}")
        0.6177
        ```

        ```pycon {.py .python linenums="1" title="Gaussian noise"}
        >>> print(f"{sample_entropy(x=data_noise, order=2):.4f}")
        2.1819
        ```

        ```pycon {.py .python linenums="1" title="Euclidean metric"}
        >>> print(f"{sample_entropy(x=data_noise, order=3, metric='euclidean'):.4f}")
        2.6806
        ```

        ```pycon {.py .python linenums="1" title="Random data"}
        >>> print(f"{sample_entropy(x=data_random):.4f}")
        2.1595
        ```

        ```pycon {.py .python linenums="1" title="Sine wave"}
        >>> print(f"{sample_entropy(x=data_sine):.4f}")
        0.1633
        ```

        ```pycon {.py .python linenums="1" title="Straight line"}
        >>> print(f"{sample_entropy(x=data_line):.4f}")
        0.0000
        ```

    ??? Question "References"
        - [Richman, J. S. et al. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049](https://journals.physiology.org/doi/epdf/10.1152/ajpheart.2000.278.6.H2039)
        - [SK-Learn: Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics)
        - [Spatial data structures and algorithms](https://docs.scipy.org/doc/scipy/tutorial/spatial.html)

    ??? Tip "See Also"
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
        - [`ts_stat_tests.algorithms.app_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.approx_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.perm_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.spectral_entropy`][ts_stat_tests.algorithms.regularity.spectral_entropy]
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
    delay: Union[int, list, np.ndarray] = 1,
    normalize: bool = False,
) -> float:
    """
    !!! note "Summary"
        Permutation entropy is a measure of the complexity or randomness of a time series. It is based on the idea of permuting the order of the values in the time series and calculating the entropy of the resulting permutation patterns.

    ???+ abstract "Details"
        The permutation entropy is a complexity measure for time-series first introduced by Bandt and Pompe in 2002.

        The formula for permutation entropy is as follows:

        $$
        PE(n) = - \\sum_{n=0}^{n!} \\times p(i) \\times \\text{log2}(p(i))
        $$

        where:

        - $n$ is the length of the sliding window,
        - $p(i)$ is the probability of the i-th ordinal pattern, and
        - the sum is taken over all possible ordinal patterns.
        - The logarithm function is base 2.

        ```
        PE(n) = - ∑ p(i) * log2(p(i))
        ```

        The calculation of permutation entropy involves the following steps:

        1. Choose the length of the sliding window ($n$).
        1. Construct the set of all possible ordinal patterns of length $n$.
        1. Compute the frequency of occurrence of each ordinal pattern in the time series.
        1. Compute the probability of occurrence of each ordinal pattern by dividing its frequency by the total number of ordinal patterns.
        1. Compute the value of permutation entropy using the formula above.

        The value of permutation entropy ranges from $0$ to $log2(n!)$, with lower values indicating higher regularity or predictability in the time series. A time series with high permutation entropy is more unpredictable or irregular, whereas a time series with low permutation entropy is more regular or predictable.

        Permutation entropy is often used in time series forecasting to assess the complexity of the data and to determine whether a time series is suitable for modeling with a particular forecasting method, such as ARIMA or neural networks. It is particularly useful for detecting nonlinear dynamics and nonstationarity in the data.

        This is the information contained in comparing $n$ consecutive values of the time series. It is clear that $0 ≤ H (n) ≤ \\text{log2}(n!)$ where the lower bound is attained for an increasing or decreasing sequence of values, and the upper bound for a completely random system where all $n!$ possible permutations appear with the same probability.

        To calculate the Permutation entropy requires the specification of a set of parameters that determine the characteristics of the time series. One of these parameters is the _embedding dimension_, which determines the number of values that are used to construct each permutation pattern.

        The embedding dimension is important in the calculation of permutation entropy because it affects the sensitivity of the measure to different patterns in the data. If the embedding dimension is too small, we may miss important patterns or variations in the time series, and the resulting permutation entropy value may not accurately reflect the underlying complexity of the signal. On the other hand, if the embedding dimension is too large, we may overfit the data and produce a permutation entropy value that is overly sensitive to noise or other random fluctuations.

        Choosing an appropriate embedding dimension is therefore crucial in ensuring that the permutation entropy calculation is robust and reliable, and captures the essential features of the time series in a meaningful way. This allows us to make more accurate and informative inferences about the behavior of the system that generated the data, and can be useful in a wide range of applications, from signal processing to data analysis and beyond.

        The embedded matrix $Y$ is created by:

        $$
        \\begin{align}
            y(i) &= [x_i,x_{i+\\text{delay}}, ...,x_{i+(\\text{order}-1) * \\text{delay}}] \\\\
            Y &= [y(1),y(2),...,y(N-(\\text{order}-1))*\\text{delay})]^T
        \\end{align}
        $$

    Params:
        x (ArrayLike):
            One-dimensional time series of shape (n_times).
        order (int, optional):
            Order of permutation entropy.<br>
            Defaults to `3`.
        delay (Union[int, list, np.ndarray], optional):
            Time delay (lag). If multiple values are passed (e.g. [1, 2, 3]), AntroPy will calculate the average permutation entropy across all these delays.<br>
            Defaults to `1`.
        normalize (bool, optional):
            If True, divide by $log2(order!)$ to normalize the entropy between $0$ and $1$. Otherwise, return the permutation entropy in bit.<br>
            Defaults to `False`.

    Returns:
        (float):
            The entropy of the data set.

    !!! success "Credit"
        - All credit goes to the [`entropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html) library.

    !!! example "Examples"

        ```pycon {.py .python linenums="1" title="Prepare data"}
        >>> import numpy as np
        >>> from sktime.datasets import load_airline
        >>> from stochastic.processes import noise as sn
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> data_airline = load_airline()
        >>> rng = np.random.default_rng(seed=42)
        >>> data_noise = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
        >>> data_random = rng.random(1000)
        >>> data_sine = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
        >>> data_line = np.arange(1000)
        ```

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> print(f"{permutation_entropy(x=data_airline):.4f}")
        2.3601
        ```

        ```pycon {.py .python linenums="1" title="Simple series"}
        >>> print(f"{permutation_entropy(x=x, order=2):.4f}")
        0.9183
        ```

        ```pycon {.py .python linenums="1" title="Normalised series"}
        >>> print(f"{permutation_entropy(x=x, order=2, normalize=True):.4f}")
        0.9183
        ```

        ```pycon {.py .python linenums="1" title="Gaussian noise"}
        >>> print(f"{permutation_entropy(x=data_noise, order=2):.4f}")
        0.9999
        ```

        ```pycon {.py .python linenums="1" title="Normalized noise"}
        >>> print(f"{permutation_entropy(x=data_noise, order=2, normalize=True):.4f}")
        0.9999
        ```

        ```pycon {.py .python linenums="1" title="Multiple delays"}
        >>> print(f"{permutation_entropy(x=data_noise, delay=[1, 2, 3], normalize=True):.4f}")
        0.9999
        ```

        ```pycon {.py .python linenums="1" title="Random data"}
        >>> print(f"{permutation_entropy(x=data_random, normalize=True):.4f}")
        0.9991
        ```
        ```pycon {.py .python linenums="1" title="Sine wave"}
        >>> print(f"{permutation_entropy(x=data_sine, normalize=True):.4f}")
        0.4463
        ```

        ```pycon {.py .python linenums="1" title="Straight line"}
        >>> print(f"{permutation_entropy(x=data_line, normalize=True):.4f}")
        0.0000
        ```

    ??? question "References"
        - [Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural complexity measure for time series." Physical review letters 88.17 (2002): 174102](http://materias.df.uba.ar/dnla2019c1/files/2019/03/permutation_entropy.pdf)

    ??? tip "See Also"
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
        - [`ts_stat_tests.algorithms.app_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.approx_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.perm_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.spectral_entropy`][ts_stat_tests.algorithms.regularity.spectral_entropy]
    """
    return cast(
        float,
        a_perm_entropy(
            x=x,
            order=order,
            delay=cast(Any, delay),
            normalize=normalize,
        ),
    )


@typechecked
def spectral_entropy(
    x: ArrayLike,
    sf: float = 1,
    method: VALID_SPECTRAL_ENTROPY_METHOD_OPTIONS = "fft",
    nperseg: Optional[int] = None,
    normalize: bool = False,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    """
    !!! note "Summary"

        Spectral entropy is a measure of the amount of complexity or unpredictability in a signal's frequency domain representation. It is used to quantify the degree of randomness or regularity in the power spectrum of a signal, which is a graphical representation of the distribution of power across different frequencies.


    ???+ abstract "Details"

        Spectral Entropy is also a measure of the distribution of power or energy in the frequency domain of a time series. It is based on the Shannon entropy, which is a measure of the uncertainty or information content of a probability distribution

        Spectral Entropy is defined to be the Shannon entropy of the power spectral density ($PSD$) of the data:

        $$
        H(x,f_s) =  -\\sum_{i=0}^{f_s/2} \\times P(i) \\times \\text{log2}(P(i))
        $$

        where:

        - $P$ is the normalised $PSD$, which is the proportion of power or energy at the $i$-th frequency band, and
        - $f_s$ is the sampling frequency.
        - The logarithm function is base 2.

        ```
        SE = - ∑ p(i) * log2(p(i))
        ```

        The calculation of spectral entropy involves the following steps:

        1. Compute the power or energy spectral density of the time series using a spectral analysis technique, such as the fast Fourier transform (FFT).
        1. Divide the frequency range of interest into non-overlapping frequency bands.
        1. Compute the proportion of power or energy in each frequency band by integrating the spectral density over the band.
        1. Compute the value of spectral entropy using the formula above.

        The value of spectral entropy ranges from $0$ to $\\text{log2}(N)$, where $N$ is the number of frequency bands. Lower values indicate a more concentrated or regular distribution of power or energy in the frequency domain, while higher values indicate a more spread-out or irregular distribution.

        Spectral entropy is often used in time series forecasting to assess the complexity of the data and to determine whether a time series is suitable for modeling with a particular forecasting method, such as spectral analysis or machine learning algorithms. It is particularly useful for detecting periodicity and cyclical patterns in the data, as well as changes in the frequency distribution over time.

        To calculate spectral entropy, we first need to compute the power spectrum of the signal using a Fourier transform or other spectral analysis method. The power spectrum represents the energy of the signal at different frequencies, and can be visualized as a graph of power versus frequency.

        Once we have the power spectrum, we can calculate the spectral entropy by applying Shannon entropy to the distribution of power across different frequencies. Shannon entropy is a measure of the amount of information or uncertainty in a probability distribution, and is given by the negative sum of the product of the probability of each frequency bin and the logarithm of that probability.

        Spectral entropy is useful in a variety of applications, such as signal processing, acoustics, and neuroscience. It can be used to characterize the complexity or regularity of a signal's frequency content, and can provide insights into the underlying processes or mechanisms that generated the signal. For example, high spectral entropy may indicate the presence of multiple sources or processes with different frequencies, while low spectral entropy may suggest the presence of a single dominant frequency or periodicity.

    Params:
        x (ArrayLike):
            `1-D` or `N-D` data array.
        sf (float, optional):
            Sampling frequency, in Hz.<br>
            Defaults to `1`.
        method (VALID_SPECTRAL_ENTROPY_METHOD_OPTIONS):
            Spectral estimation method:<br>
            - `'fft'`: Fourier Transformation ([`scipy.signal.periodogram()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html#scipy.signal.periodogram))<br>
            - `'welch'`: Welch periodogram ([`scipy.signal.welch()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch))<br>
            Defaults to `"fft"`.
        nperseg (Optional[int]):
            Length of each FFT segment for Welch method. If `None`, uses `scipy`'s default of 256 samples.<br>
            Defaults to `None`.
        normalize (bool, optional):
            If `True`, divide by $log2(psd.size)$ to normalize the spectral entropy to be between $0$ and $1$. Otherwise, return the spectral entropy in bit.<br>
            Defaults to `False`.
        axis (int, optional):
            The axis along which the entropy is calculated. Default is the last axis.<br>
            Defaults to `-1`.

    Returns:
        (float):
            Spectral Entropy score.

    !!! Success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ???+ Example "Examples"

        ```pycon {.py .python linenums="1" title="Prepare data"}
        >>> import numpy as np
        >>> from sktime.datasets import load_airline
        >>> from stochastic.processes import noise as sn
        >>> sf, dur = 100, 4
        >>> N = sf * dur
        >>> data_time = np.arange(N)
        >>> data_sine = np.sin(2 * np.pi * 1 * data_time)
        >>> rng = np.random.default_rng(seed=42)
        >>> data_noise = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
        >>> data_2d = rng.normal(size=(4, 3000))
        >>> data_airline = load_airline()
        ```

        ```pycon {.py .python linenums="1"  title="Basic usage"}
        >>> print(f"{spectral_entropy(x=data_airline, sf=12):.4f}")
        2.6538
        ```

        ```pycon {.py .python linenums="1" title="Sine wave"}
        >>> print(f"{spectral_entropy(x=data_sine, sf=100, method='fft'):.4f}")
        6.2329
        ```

        ```pycon {.py .python linenums="1"  title="Welch method"}
        >>> print(f"{spectral_entropy(x=data_sine, sf=100, method='welch'):.4f}")
        1.2924
        ```

        ```pycon {.py .python linenums="1"  title="Normalised calculation"}
        >>> print(f"{spectral_entropy(x=data_sine, sf=100, method='welch', normalize=True):.4f}")
        0.9956
        ```

        ```pycon {.py .python linenums="1"  title="2D data"}
        >>> print(spectral_entropy(x=data_2d, sf=100, normalize=True).tolist())
        [0.9426, 0.9382, 0.9410, 0.9376]
        ```

        ```pycon {.py .python linenums="1"  title="Gaussian noise"}
        >>> print(f"{spectral_entropy(x=data_noise, sf=100, normalize=True):.4f}")
        0.9505
        ```

    ??? Question "References"
        - Inouye, T. et al. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum. Electroencephalography and clinical neurophysiology, 79(3), 204-210.
        - https://en.wikipedia.org/wiki/Spectral_density
        - https://en.wikipedia.org/wiki/Welch%27s_method

    ??? Tip "See Also"
        - [`antropy.app_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.app_entropy.html)
        - [`antropy.sample_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html)
        - [`antropy.perm_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.perm_entropy.html)
        - [`antropy.spectral_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html)
        - [`ts_stat_tests.algorithms.app_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.approx_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.perm_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.spectral_entropy`][ts_stat_tests.algorithms.regularity.spectral_entropy]
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
    """
    !!! note "Summary"
        SVD entropy is a measure of the complexity or randomness of a time series based on Singular Value Decomposition (SVD).

    ???+ abstract "Details"
        SVD entropy is calculated by first embedding the time series into a matrix, then performing SVD on that matrix to obtain the singular values. The entropy is then calculated from the normalized singular values.

    Params:
        x (ArrayLike):
            One-dimensional time series of shape (n_times).
        order (int, optional):
            Order of the SVD entropy (embedding dimension).<br>
            Defaults to `3`.
        delay (int, optional):
            Time delay (lag).<br>
            Defaults to `1`.
        normalize (bool, optional):
            If True, divide by $log2(order!)$ to normalize the entropy between $0$ and $1$.<br>
            Defaults to `False`.

    Returns:
        (float):
            The SVD entropy of the data set.

    !!! Success "Credit"
        All credit goes to the [`AntroPy`](https://raphaelvallat.com/antropy/) library.

    ??? Tip "See Also"
        - [`antropy.svd_entropy`](https://raphaelvallat.com/antropy/build/html/generated/antropy.svd_entropy.html)
        - [`ts_stat_tests.algorithms.approx_entropy`][ts_stat_tests.algorithms.regularity.approx_entropy]
        - [`ts_stat_tests.algorithms.sample_entropy`][ts_stat_tests.algorithms.regularity.sample_entropy]
        - [`ts_stat_tests.algorithms.perm_entropy`][ts_stat_tests.algorithms.regularity.permutation_entropy]
        - [`ts_stat_tests.algorithms.spectral_entropy`][ts_stat_tests.algorithms.regularity.spectral_entropy]
    """
    return a_svd_entropy(
        x=x,
        order=order,
        delay=delay,
        normalize=normalize,
    )
