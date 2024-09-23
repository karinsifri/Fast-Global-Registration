import numpy as np

from src.utils.transforms import apply_homogenous_transformation


def pairwise_objective_2d(xi: tuple[float, float, float], p: np.ndarray, q: np.ndarray, t: np.ndarray,
                          l_weights: np.ndarray) -> np.ndarray:
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
    trans: np.ndarray = reconstruct_transformation_2d(t, xi)
    dists: np.ndarray = np.linalg.norm(p - apply_homogenous_transformation(points=q, transformation_matrix=trans),
                                       axis=1)
    return l_weights * (dists ** 2)


def reconstruct_transformation_2d(t: np.ndarray, xi: tuple[float, float, float]) -> np.ndarray:
    """
    An adaptation of the reconstruct_transformation function to fit 2-dimensional data

    Args:
        t (np.ndarray): the previous transformation matrix
        xi (tuple): the linearizarion of the new transformation matrix from the previous transformation matrix

    Returns:
        t_new (np.ndarray): the constructed transformation matrix
    """
    alpha, a, b = xi
    mat = np.array([[np.cos(alpha), -np.sin(alpha), a],
                    [np.sin(alpha), np.cos(alpha), b],
                    [0, 0, 1]])
    return mat @ t
