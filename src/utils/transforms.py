import numpy as np
from scipy.linalg import polar


def apply_homogenous_transformation(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a homogenous transformation to a list of points

    Args:
        points: an array of shape (N, dim) containing the points coordinates
        transformation_matrix: a matrix of shape (dim+1, dim+1) containing the transformation

    Returns:
        an array of shape (N, dim) containing the coordinates of the transformed points
    """
    # shape validations
    if len(points.shape) != 2 or len(transformation_matrix.shape) != 2:
        raise ValueError("Both points and transformation should be 2 dimensional arrays")
    num_points, dim = points.shape
    if transformation_matrix.shape[0] != transformation_matrix.shape[1]:
        raise ValueError("Transformation matrix should be squared")
    if transformation_matrix.shape[0] != dim + 1:
        raise ValueError("Transformation matrix should have exactly one more dimension then the points")

    points = np.column_stack([points, np.ones(num_points)])
    points = transformation_matrix @ points.T
    points = points[:-1] / points[-1]
    return points.T


def project_to_rigid(t: np.ndarray) -> np.ndarray:
    """
    Projects a transformation matrix d_new to SE(n), ensuring it's a valid rigid body transformation.

    Args:
        t (ndarray): The transformation matrix of shape (n+1, n+1).

    Returns:
        t_rigid (ndarray): The projected SE(n) matrix.
    """
    if t.shape[0] != t.shape[1]:
        raise ValueError("Input matrix must be square.")
    n = t.shape[0] - 1
    if n < 1:
        raise ValueError("Invalid transformation matrix dimension.")

    # Project the rotational part to SO(n) using polar decomposition
    rot, _ = polar(t[:n, :n])

    # Ensure det(R_orthogonal) = 1 for proper rotation
    rot[:, -1] *= np.sign(np.linalg.det(rot))

    # Construct the SE(n) matrix by copying over translation and corrected rotation
    t_rigid = np.eye(n + 1)
    t_rigid[:n, :n] = rot
    t_rigid[:n, n] = t[:n, n]

    return t_rigid
