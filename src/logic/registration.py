import numpy as np
from open3d.cpu.pybind.geometry import PointCloud

from src.logic.correspondence import find_point_correspondence
from src.logic.optimization import optimization_step

MAX_ITERATIONS = 64  # TODO: move to consts


def pairwise_registration(pcd_p: PointCloud, pcd_q: PointCloud) -> np.ndarray:
    # TODO: write docstring
    points_p, points_q = find_point_correspondence(pcd_p, pcd_q)
    points_p, points_q = normalize_points(points_p, points_q)

    t = np.identity(4)
    mu = 1

    for i in range(MAX_ITERATIONS):
        t = optimization_step(points_p, points_q, t, mu)

        if i % 4 == 3:  # every 4 iterations
            mu /= 2

    return t


def normalize_points(points_p: np.ndarray, points_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_p = np.mean(points_p, axis=0)
    mean_q = np.mean(points_q, axis=0)
    scale = np.linalg.norm(points_p - mean_p, axis=1).max()
    new_p = (points_p - mean_p) / scale
    new_q = (points_q - mean_q) / scale
    return new_p, new_q
