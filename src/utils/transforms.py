import numpy as np


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
