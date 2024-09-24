import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud

from src.consts import KDTREE_MAX_NN

VOXEL_SIZE = 0.05


def run_fgr(pcd_p: PointCloud, pcd_q: PointCloud) -> np.ndarray:
    """
    Run the implemented Fast-Global-Registration algorith in open3d.

    Args:
        pcd_p: a point-cloud object
        pcd_q: a point-cloud object that we want to align to pcd_p

    Returns:
        transformation: the transformation that aligns pcd_q to pcd_p
    """
    pcd_p = pcd_p.voxel_down_sample(VOXEL_SIZE)
    pcd_q = pcd_q.voxel_down_sample(VOXEL_SIZE)

    fpfh_p = o3d.pipelines.registration.compute_fpfh_feature(pcd_p,
                                                             o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5,
                                                                                                  max_nn=KDTREE_MAX_NN))
    fpfh_q = o3d.pipelines.registration.compute_fpfh_feature(pcd_q,
                                                             o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5,
                                                                                                  max_nn=KDTREE_MAX_NN))

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd_p, pcd_q, fpfh_p, fpfh_q,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=VOXEL_SIZE*1.5)
    )

    return result.transformation
