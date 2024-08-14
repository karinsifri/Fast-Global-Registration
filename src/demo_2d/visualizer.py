from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def visualize_surfaces(surface1, surface2, points1, points2, l):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the surfaces
    ax.plot(surface1[:, 0], surface1[:, 1], label='Surface 1', color='blue')
    ax.plot(surface2[:, 0], surface2[:, 1], label='Surface 2', color='red')

    # Scatter the points on both surfaces
    ax.scatter(points1[:, 0], points1[:, 1], color='blue', s=50, label='Points on Surface 1', edgecolor='black')
    ax.scatter(points2[:, 0], points2[:, 1], color='red', s=50, label='Points on Surface 2', edgecolor='black')

    # Normalize the l values to [0, 1] for the colormap
    norm = mcolors.Normalize(vmin=np.min(l), vmax=np.max(l))

    # Create a colormap (using RdYlGn to represent good/bad scale)
    cmap = plt.cm.RdYlGn

    # Plot the connections with colors representing the l values
    for i in range(len(points1)):
        ax.plot([points1[i, 0], points2[i, 0]], [points1[i, 1], points2[i, 1]],
                color=cmap(norm(l[i])), alpha=0.6)

    # Add a colorbar to show the scale of l values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Connection Strength (l)', rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Surface Visualization with Points and Connections')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
