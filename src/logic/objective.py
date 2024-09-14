import numpy as np

from src.utils.linearization import reconstruct_transformation
from src.utils.transforms import apply_homogenous_transformation


def pairwise_objective(xi: tuple[float, float, float, float, float, float],
                       p: np.ndarray, q: np.ndarray, t: np.ndarray, l_weights: np.ndarray) -> np.ndarray:
    """
    TODO: write docstring

    Args:
        xi:
        p:
        q:
        t:
        l_weights:

    Returns:

    """
    trans = reconstruct_transformation(t, xi)
    dists: np.ndarray = np.linalg.norm(p - apply_homogenous_transformation(points=q, transformation_matrix=trans),
                                       axis=1)
    return l_weights * (dists ** 2)
