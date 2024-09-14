import numpy as np
from open3d.cpu.pybind.geometry import PointCloud

from src.logic.correspondence import find_point_correspondence
from src.logic.optimization import optimization_step

MAX_ITERATIONS = 64  # TODO: move to consts


def pairwise_registration(pcd_p: PointCloud, pcd_q: PointCloud) -> np.ndarray:
    points_p, points_q = find_point_correspondence(pcd_p, pcd_q)
    # TODO: write docstring

    t = np.identity(4)
    mu = 100  # TODO: set to be the max diameter of the point clouds

    for i in range(MAX_ITERATIONS):
        t = optimization_step(points_p, points_q, t, mu)

        if i % 4 == 3:  # every 4 iterations
            mu /= 2

    return t
