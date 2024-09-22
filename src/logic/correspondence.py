import numpy as np
from sklearn.neighbors import KDTree
from open3d.cpu.pybind.pipelines.registration import compute_fpfh_feature
from open3d.cpu.pybind.geometry import PointCloud, KDTreeSearchParamHybrid

from src.consts import KDTREE_RADIUS, KDTREE_MAX_NN, TUPLE_TEST_TRIALS_SCALE, TUPLE_TEST_MAX_CORRESPONDENCES, TAU


def find_point_correspondence(pcd_p: PointCloud, pcd_q: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Point Correspondences according to section 3.3 of the paper

    Args:
        pcd_p (PointCloud): the point cloud of the first object {P}
        pcd_q (PointCloud): the point cloud of the second object {Q}

    Returns:
        kappa (tuple[ndarray, ndarray]): a tuple containing two list of the matching points from each of the point
            clouds
    """
    kappa_1 = collect_all_correspondences(pcd_p, pcd_q)

    kappa_2 = reciprocity_test(kappa_1)

    kappa_3 = tuple_test(kappa_2, pcd_p, pcd_q)

    kappa = get_matches_to_points(kappa_3, pcd_p, pcd_q)

    return kappa


def collect_all_correspondences(pcd_p: PointCloud, pcd_q: PointCloud) -> np.ndarray:
    """
    Get all correspondences by using nearest neighbor queries for each point in the point clouds

    Args:
        pcd_p (PointCloud): the point cloud of the first object {P}
        pcd_q (PointCloud): the point cloud of the second object {Q}

    Returns:
        kappa1 (np.ndarray): a list of point indices describing all the correspondences
    """
    fpfh_p = compute_fpfh_feature(pcd_p, KDTreeSearchParamHybrid(radius=KDTREE_RADIUS, max_nn=KDTREE_MAX_NN))
    fpfh_q = compute_fpfh_feature(pcd_q, KDTreeSearchParamHybrid(radius=KDTREE_RADIUS, max_nn=KDTREE_MAX_NN))

    f_p = np.asarray(fpfh_p.data, dtype=np.float32).T
    f_q = np.asarray(fpfh_q.data, dtype=np.float32).T

    kdtree_p = KDTree(f_p)
    kdtree_q = KDTree(f_q)

    _, indices_q = kdtree_q.query(f_p, k=1)
    _, indices_p = kdtree_p.query(f_q, k=1)

    matches1 = np.column_stack([np.expand_dims(np.arange(len(f_p)), axis=1), indices_q])
    matches2 = np.column_stack([indices_p, np.expand_dims(np.arange(len(f_q)), axis=1)])

    kappa1 = np.row_stack([matches1, matches2])
    return kappa1


def reciprocity_test(kappa_1: np.ndarray) -> np.ndarray:
    """
    Compute the reciprocity test for the collected matches.

    A match is reciprocal if and only if  F(p) is the nearest neighbor of F(q) among F(P) and F(q) is the nearest
    neighbor of F(p) among F(Q). therefore, it is inserted into kappa_1 twice, and we can return only the matches that
    appear twice in kappa_1.

    Args:
        kappa_1 (np.ndarray): a list of point indices describing the all correspondences

    Returns:
        kappa_2 (np.ndarray): the indices of the matches that pass the reciprocity test
    """
    unique, counts = np.unique(kappa_1, axis=0, return_counts=True)
    kappa_2 = unique[counts == 2]
    return kappa_2


def tuple_test(kappa_2: np.ndarray, pcd_p: PointCloud, pcd_q: PointCloud) -> np.ndarray:
    """
    Return point matches that pass the tuple test

    Args:
        kappa_2 (np.ndarray): the indices of the matches that pass the reciprocity test
        pcd_p (PointCloud): the point cloud of the first object {P}
        pcd_q (PointCloud): the point cloud of the second object {Q}

    Returns:
        kappa_3 (np.ndarray): indices of matches that pass the tuple test
    """
    # get random tuples
    tuples = np.random.randint(len(kappa_2), size=(len(kappa_2) * TUPLE_TEST_TRIALS_SCALE, 3))

    # make sure that there are no matches that appear more than once in the tuple
    all_different = np.logical_and(tuples[:, 0] != tuples[:, 1],
                                   np.logical_and(tuples[:, 0] != tuples[:, 2], tuples[:, 1] != tuples[:, 2]))
    tuples = tuples[all_different]

    points_p = np.asarray(pcd_p.points)
    points_q = np.asarray(pcd_q.points)

    # get the points for each of the tuples
    p_i = points_p[kappa_2[tuples[:, 0]][:, 0]]
    q_i = points_q[kappa_2[tuples[:, 0]][:, 1]]
    p_j = points_p[kappa_2[tuples[:, 1]][:, 0]]
    q_j = points_q[kappa_2[tuples[:, 1]][:, 1]]
    p_k = points_p[kappa_2[tuples[:, 2]][:, 0]]
    q_k = points_q[kappa_2[tuples[:, 2]][:, 1]]

    # compute point distance ratio
    first = np.linalg.norm(p_i - p_j, axis=1) / np.linalg.norm(q_i - q_j, axis=1)
    second = np.linalg.norm(p_i - p_k, axis=1) / np.linalg.norm(q_i - q_k, axis=1)
    third = np.linalg.norm(p_j - p_k, axis=1) / np.linalg.norm(q_j - q_k, axis=1)

    passed_tuple_test = np.logical_and(
        np.logical_and(
            np.logical_and(first > TAU, first < 1 / TAU),
            np.logical_and(second > TAU, second < 1 / TAU)
        ),
        np.logical_and(third > TAU, third < 1 / TAU)
    )

    # get the first (unique) matches that pass the tuple test
    kappa_3 = kappa_2[np.unique(tuples[np.flatnonzero(passed_tuple_test)[:TUPLE_TEST_MAX_CORRESPONDENCES]].flatten())]

    return kappa_3


def get_matches_to_points(matches: np.ndarray, pcd_p: PointCloud, pcd_q: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the match indices back to points

    Args:
        matches (np.ndarray): a list of point indices describing the matches
        pcd_p (PointCloud): the point cloud of the first object {P}
        pcd_q (PointCloud): the point cloud of the second object {Q}

    Returns:
        kappa (tuple[ndarray, ndarray]): a tuple containing two list of the matching points from each of the point
            clouds
    """
    points_p = np.asarray(pcd_p.points)
    points_q = np.asarray(pcd_q.points)

    return points_p[matches[:, 0]], points_q[matches[:, 1]]
