import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from scipy.spatial import ConvexHull, distance_matrix


def get_diameter(pcd: PointCloud) -> float:
    # TODO: add docstring
    points = np.asarray(pcd.points)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_dist_matrix = distance_matrix(hull_points, hull_points)
    approx_diameter = np.max(hull_dist_matrix)
    return approx_diameter
