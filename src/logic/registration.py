import numpy as np
from open3d.cpu.pybind.geometry import PointCloud

from src.consts import MAX_ITERATIONS
from src.logic.optimization import optimization_step
from src.logic.correspondence import find_point_correspondence


def fast_global_registration(pcd_p: PointCloud, pcd_q: PointCloud) -> np.ndarray:
    """
    Run the pairwise Fast-Global_Registration algorithm as described in the paper

    Args:
        pcd_p (PointCloud): a point-cloud object
        pcd_q (PointCloud): a point-cloud object that we want to align to pcd_p

    Returns:
        t (ndarray): a rigid transformation matrix (4, 4) that aligns pcd_q to in the same coordinate system as pcd_p
    """
    points_p, points_q = find_point_correspondence(pcd_p, pcd_q)

    t = np.identity(4)
    mu = 1

    for i in range(MAX_ITERATIONS):
        t = optimization_step(points_p, points_q, t, mu)

        if i % 4 == 3:  # every 4 iterations
            mu /= 2

    return t
