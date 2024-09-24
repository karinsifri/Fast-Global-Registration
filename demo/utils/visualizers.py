from copy import deepcopy

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


def draw_registration_result(source: PointCloud, target: PointCloud, transformation: np.ndarray) -> None:
    """
    Visualizes the registration result with the object centered in the view and a white background.

    Args:
        source: the point cloud of the source object
        target: the point cloud of the target object
        transformation: The 4x4 transformation matrix to align source to target

    Returns:

    """
    """
    Visualizes the registration result with the object centered in the view and a white background.

    Parameters:
    - pcd1 (o3d.geometry.PointCloud): The pcd1 point cloud.
    - pcd2(o3d.geometry.PointCloud): The pcd2 point cloud.
    - transformation (numpy.ndarray): The 4x4 transformation matrix to align source to target.
    """
    # Deep copy to avoid modifying the original point clouds
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)

    # Apply colors for differentiation
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange for pcd1
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue for pcd2

    # Apply the transformation to the source
    source_temp.transform(transformation)

    # Combine both point clouds for computing the center
    combined = source_temp + target_temp

    # Compute the axis-aligned bounding box of the combined point clouds
    bbox = combined.get_axis_aligned_bounding_box()

    # Calculate the center of the bounding box
    center = bbox.get_center()

    # Initialize the Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    # Add the transformed source and target point clouds
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)

    # Set the background color to white
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background
    opt.point_size = 2.0

    # Get the view control and set camera parameters to center the object
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    # Update renderer to apply the background change
    vis.update_renderer()

    # Run the visualizer
    vis.run()
    vis.destroy_window()
