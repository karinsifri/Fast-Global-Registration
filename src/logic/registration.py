import numpy as np
from open3d.cpu.pybind.geometry import PointCloud

from src.logic.correspondence import find_point_correspondence
from src.logic.optimization import optimization_step

MAX_ITERATIONS = 64  # TODO: move to consts


def pairwise_registration(pcd_p: PointCloud, pcd_q: PointCloud) -> np.ndarray:
    # TODO: write docstring
    pcd_p, pcd_q = normalize_points(pcd_p, pcd_q)
    points_p, points_q = find_point_correspondence(pcd_p, pcd_q)

    t = np.identity(4)
    mu = 1

    for i in range(MAX_ITERATIONS):
        t = optimization_step(points_p, points_q, t, mu)

        if i % 4 == 3:  # every 4 iterations
            mu /= 2

    return t


def normalize_points(points_p: PointCloud, points_q: PointCloud) -> tuple[PointCloud, PointCloud]:
    mean_p = np.asarray(points_p.points).mean(axis=0)
    mean_q = np.asarray(points_q.points).mean(axis=0)
    scale = np.linalg.norm(np.asarray(points_p.points) - mean_p, axis=1).max()
    transform = np.identity(4)
    transform[:3, :3] /= scale
    transform_p = transform
    transform_p[:3, 3] -= mean_p
    transform_q = transform
    transform_q[:3, 3] -= mean_q
    new_p = points_p.transform(transform_p)
    new_q = points_q.transform(transform_q)
    return new_p, new_q
