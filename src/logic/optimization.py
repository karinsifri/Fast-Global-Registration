import numpy as np
from scipy.optimize import least_squares

from src.logic.objective import pairwise_objective
from src.utils.linearization import reconstruct_transformation
from src.utils.transforms import apply_homogenous_transformation, project_to_rigid


def optimization_step(p: np.ndarray, q: np.ndarray, t: np.ndarray, mu: float) -> np.ndarray:
    """
    Updates the transformation matrix according to the optimization step

    Args:
        p (np.ndarray): the matching points from the first point cloud
        q (np.ndarray): the matching points from the second point cloud
        t (np.ndarray): the current value of the transformation matching the first point cloud to the second point cloud
        mu (float): the Geman-McClure scale (as described in section 3.1)

    Returns:
        t (np.ndarray): the updated transformation matrix calculated after the optimization step
    """
    # calculate the l weights according to the partial derivatives (formula 6)
    dists: np.ndarray = np.linalg.norm(p - apply_homogenous_transformation(points=q, transformation_matrix=t), axis=1)
    l_weights: np.ndarray = (mu / (mu + dists ** 2)) ** 2

    # find the new transformation using Levenberg-Marquardt algorithm
    xi_0 = 0, 0, 0, 0, 0, 0
    result = least_squares(pairwise_objective, xi_0, args=(p, q, t, l_weights), method='lm')

    t_new: np.ndarray = reconstruct_transformation(t, result.x)

    return project_to_rigid(t_new)
