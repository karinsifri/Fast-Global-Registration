import numpy as np
from open3d.cpu.pybind.geometry import PointCloud


def normalize_point_clouds(source_cloud: PointCloud, target_cloud: PointCloud) -> tuple[PointCloud, PointCloud]:
    """
    Normalize two point clouds such that their centroids are at the origin and their scales are uniform.

    Args:
        source_cloud (PointCloud): The first point cloud to be normalized.
        target_cloud (PointCloud): The second point cloud to be normalized.

    Returns:
        tuple[PointCloud, PointCloud]: The normalized source and target point clouds.
    """
    # Convert point clouds to NumPy arrays for easier manipulation
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)

    # Compute centroids (mean) of each point cloud
    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    # Compute the scaling factor using the maximum distance from the source centroid
    scale = np.linalg.norm(source_points - source_centroid, axis=1).max()

    # Create a transformation matrix with uniform scaling
    scaling_transform = np.identity(4)
    scaling_transform[:3, :3] /= scale

    # Apply transformations to move centroids to origin
    source_transform = scaling_transform.copy()
    source_transform[:3, 3] -= source_centroid

    target_transform = scaling_transform.copy()
    target_transform[:3, 3] -= target_centroid

    # Transform the point clouds
    normalized_source_cloud = source_cloud.transform(source_transform)
    normalized_target_cloud = target_cloud.transform(target_transform)

    return normalized_source_cloud, normalized_target_cloud
