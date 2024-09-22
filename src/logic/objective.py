import numpy as np

from src.utils.linearization import reconstruct_transformation
from src.utils.transforms import apply_homogenous_transformation


def pairwise_objective(xi: tuple[float, float, float, float, float, float],
                       p: np.ndarray, q: np.ndarray, t: np.ndarray, l_weights: np.ndarray) -> np.ndarray:
    """
    The pairwise objective function to optimize when L is fixed

    Args:
        xi (tuple): the linearization of the new transformation, the parameters to optimize
        p (np.ndarray): the matching points from the first point cloud
        q (np.ndarray): the matching points from the second point cloud
        t (np.ndarray): the current value of the transformation matching the first point cloud to the second point cloud
        l_weights (np.ndarray): the weights of the matches between the point clouds

    Returns:
        np.ndarray: weighted L2 penalties of distances between the point correspondences
    """
    trans: np.ndarray = reconstruct_transformation(t, xi)
    dists: np.ndarray = np.linalg.norm(p - apply_homogenous_transformation(points=q, transformation_matrix=trans),
                                       axis=1)
    return l_weights * (dists ** 2)
