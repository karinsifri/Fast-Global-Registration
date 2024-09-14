import numpy as np
from scipy.optimize import least_squares

from src.logic.objective import pairwise_objective
from src.utils.linearization import xi_to_mat
from src.utils.transforms import apply_homogenous_transformation, project_to_rigid


def optimization_step(p: np.ndarray, q: np.ndarray, t: np.ndarray, mu: float) -> np.ndarray:
    """
    TODO: write docstring

    Args:
        p:
        q:
        t:
        mu:

    Returns:

    """
    dists: np.ndarray = np.linalg.norm(p - apply_homogenous_transformation(points=q, transformation_matrix=t), axis=1)
    l_weights: np.ndarray = (mu / (mu + dists ** 2)) ** 2

    xi_0 = 0, 0, 0, 0, 0, 0
    result = least_squares(pairwise_objective, xi_0, args=(p, q, t, l_weights), method='lm')

    t_new: np.ndarray = xi_to_mat(*result.x) @ t

    return project_to_rigid(t_new)
