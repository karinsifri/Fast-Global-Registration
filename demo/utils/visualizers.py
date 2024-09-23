import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def visualize_step_2d(spline1: np.ndarray, spline2: np.ndarray, points1: np.ndarray, points2: np.ndarray, l_weights: np.ndarray, step: int) -> None:
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
    cmap = plt.cm.Greens

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
