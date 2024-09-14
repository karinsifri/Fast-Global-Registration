import numpy as np


def reconstruct_transformation(t: np.ndarray, xi: tuple[float, float, float, float, float, float]) -> np.ndarray:
    """
    Given a linearization of a transformation matrix from its previous value, reconstruct the new transformation matrix

    Args:
        t (np.ndarray): the previous transformation matrix
        xi (tuple): the linearizarion of the new transformation matrix from the previous transformation matrix

    Returns:
        t_new (np.ndarray): the constructed transformation matrix
    """
    alpha, beta, gamma, a, b, c = xi
    mat = np.array([[1, -gamma, beta, a],
                    [gamma, 1, -alpha, b],
                    [-beta, alpha, 1, c],
                    [0, 0, 0, 1]])
    return mat @ t
