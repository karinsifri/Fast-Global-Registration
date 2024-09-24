import numpy as np
import open3d as o3d
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from open3d.cpu.pybind.geometry import PointCloud

from src.logic.correspondence import get_matches_to_points


def visualize_step_2d(spline1: np.ndarray, spline2: np.ndarray, points1: np.ndarray, points2: np.ndarray,
                      l_weights: np.ndarray, step: int) -> None:
    """
    Visualize a 2d optimization step

    Args:
        spline1: an array containing the linestring points1 were sampled from
        spline2: an array containing the linestring points2 were sampled from
        points1: the matching points sampled from spline1
        points2: the matching points sampled from spline2
        l_weights: the weights of the point matches
        step: the step number
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the surfaces
    ax.plot(spline1[:, 0], spline1[:, 1], label='Spline 1', color='blue')
    ax.plot(spline2[:, 0], spline2[:, 1], label='Spline 2', color='red')

    # Scatter the points on both surfaces
    ax.scatter(points1[:, 0], points1[:, 1], color='blue', s=50, label='Points on Spline 1', edgecolor='black')
    ax.scatter(points2[:, 0], points2[:, 1], color='red', s=50, label='Points on Spline 2', edgecolor='black')

    # Normalize the l values to [0, 1] for the colormap
    norm = mcolors.Normalize(vmin=np.min(l_weights), vmax=np.max(l_weights))

    # Create a colormap (using RdYlGn to represent good/bad scale)
    cmap = plt.cm.coolwarm_r

    # Plot the connections with colors representing the l values
    for i in range(len(points1)):
        ax.plot([points1[i, 0], points2[i, 0]], [points1[i, 1], points2[i, 1]],
                color=cmap(norm(l_weights[i])), alpha=0.6)

    # Add a colorbar to show the scale of l values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Connection Strength (l)', rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Step {step}')

    # Add a legend
    ax.legend()

    # Set aspect ratio
    ax.set_aspect("equal")

    # Show the plot
    plt.show()


def visualize_correspondences(pcd_p: PointCloud, pcd_q: PointCloud, matches: np.ndarray, title: str) -> None:
    """
    Visualizes the point correspondences using lines between matched points.

    Args:
        pcd_p: the point cloud of the first object {P}
        pcd_q: the point cloud of the second object {Q}
        matches: indices of matching point between pcd_p and pcd_q
        title: a name for the visualisation window
    """
    if len(matches) == 0:
        print(f"No matches to visualize in {title}")
        return

    lines = [[i, i + len(matches)] for i in range(len(matches))]  # Create line pairs

    matched_points_p, matched_points_q = get_matches_to_points(matches=matches, pcd_p=pcd_p, pcd_q=pcd_q)

    combined_points = np.vstack((matched_points_p, matched_points_q))

    # Create lineset to draw correspondences
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(combined_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([0, 1, 0])

    # Visualize point clouds and correspondences
    pcd_p.paint_uniform_color([1, 0, 0])  # Red for P
    pcd_q.paint_uniform_color([0, 0, 1])  # Blue for Q

    o3d.visualization.draw_geometries([pcd_p, pcd_q, line_set], window_name=title)
