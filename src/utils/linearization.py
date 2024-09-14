import numpy as np


def xi_to_mat(alpha: float, beta: float, gamma: float, a: float, b: float, c: float) -> np.ndarray:
    """
    TODO: write docstring

    Args:
        alpha:
        beta:
        gamma:
        a:
        b:
        c:

    Returns:

    """
    mat = np.array([[1, -gamma, beta, a],
                    [gamma, 1, -alpha, b],
                    [-beta, alpha, 1, c],
                    [0, 0, 0, 1]])
    return mat
