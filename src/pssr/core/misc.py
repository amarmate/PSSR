import numpy as np


def sample_without_replacement(
    rng: np.random.Generator,
    s: int,
    m: int,
    n_cases: int,
) -> np.ndarray:
    """
    Generate s independent uniform samples without replacement of size n_cases
    from the index set {0, ..., m-1}.

    The function first assigns i.i.d. random scores to each of the m indices
    and, for each sample, selects the n_cases indices with the smallest scores.
    This yields a uniformly chosen n_cases-subset without duplicates. It then
    randomly shuffles each row so that the order of indices within each sample
    is also uniform.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used for all sampling.
    s : int
        Number of independent samples (rows) to generate.
    m : int
        Total number of possible indices (0 through m-1).
    n_cases : int
        Number of indices (columns) to draw per sample (0 < n_cases <= m).

    Returns
    -------
    case_order : ndarray of shape (s, n_cases)
        For each of the s samples, a uniformly random subset of n_cases distinct
        indices from {0, ..., m-1}, sampled without replacement and with a
        uniformly random permutation within each row.
    """

    scores = rng.random((s, m))

    case_order = np.argpartition(scores,
                                 kth=n_cases - 1,
                                 axis=1)[:, :n_cases]

    perm_scores = rng.random((s, n_cases))
    perm_idx = np.argsort(perm_scores, axis=1)
    return np.take_along_axis(case_order, perm_idx, axis=1)
