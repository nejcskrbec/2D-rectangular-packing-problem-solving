import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np

from shapely.geometry import Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString, Point
from matplotlib.gridspec import GridSpec
from math import sqrt

def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def is_point_on_line_vec(points, line_start, line_end):
    """Vectorized function to check if points lie on a line segment."""

    x, y = points.T
    x1, y1 = line_start.T
    x2, y2 = line_end.T

    # Create a grid for each combination of points and line segments
    x = x[:, None]
    y = y[:, None]

    # Collinearity condition
    collinear = (y2 - y1) * (x - x1) == (y - y1) * (x2 - x1)

    # Bounding box condition
    within_box = np.logical_and.reduce([
        np.minimum(x1, x2) <= x, x <= np.maximum(x1, x2),
        np.minimum(y1, y2) <= y, y <= np.maximum(y1, y2)
    ])

    return np.logical_and(collinear, within_box).any(axis=1)

def extract_intersection_points(intersection):
    if intersection.is_empty:
        return set()

    points = set()
    if isinstance(intersection, (LineString, Point)):
        points.update(intersection.coords)

    elif isinstance(intersection, MultiLineString):
        for line_string in intersection.geoms:
            points.update(line_string.coords)

    return points


def find_intersections(line, selected_point, knapsack):
    items_combined_shape = knapsack.get_items_combined_shape()
    polygons = [items_combined_shape] if not isinstance(items_combined_shape, MultiPolygon) else list(items_combined_shape.geoms)
    polygons.append(knapsack.shape)

    knapsack_items_intersection = MultiPolygon()
    for polygon in polygons:
        intersection = line.intersection(polygon)
        if not intersection.is_empty:
            knapsack_items_intersection = knapsack_items_intersection.union(intersection)

    intersection_points = extract_intersection_points(knapsack_items_intersection)
    # Filter out the original selected_point
    intersection_points.discard(tuple(selected_point))

    return sorted(intersection_points, key=lambda p: (p[0] - selected_point[0])**2 + (p[1] - selected_point[1])**2)


def project_up(selected_point, knapsack):
    _, _, _, maxy = knapsack.get_bounds()
    vertical_line_up = LineString([(selected_point[0], selected_point[1]), (selected_point[0], maxy)])
    return find_intersections(vertical_line_up, selected_point, knapsack)


def project_down(selected_point, knapsack):
    _, miny, _, _ = knapsack.get_bounds()
    vertical_line_down = LineString([(selected_point[0], selected_point[1]), (selected_point[0], miny)])
    return find_intersections(vertical_line_down, selected_point, knapsack)


def project_left(selected_point, knapsack):
    minx, _, _, _ = knapsack.get_bounds()
    horizontal_line_left = LineString([(selected_point[0], selected_point[1]), (minx, selected_point[1])])
    return find_intersections(horizontal_line_left, selected_point, knapsack)


def project_right(selected_point, knapsack):
    _, _, maxx, _ = knapsack.get_bounds()
    horizontal_line_right = LineString([(selected_point[0], selected_point[1]), (maxx, selected_point[1])])
    return find_intersections(horizontal_line_right, selected_point, knapsack)


def visualize_knapsack_and_items(knapsack, items, title):
    # Create a plot to visualize the knapsack and items
    fig, ax = plt.subplots()

    # Plot the knapsack
    ax.plot(*knapsack.shape.exterior.xy, color='blue', linewidth=2)

    # Adjust viridis colormap to use only the middle third
    original_cmap = plt.cm.viridis
    colors = original_cmap(np.linspace(0.33, 0.66, 256))  # Extract the middle third of the viridis spectrum
    cmap = mcolors.LinearSegmentedColormap.from_list("viridis_middle", colors)

    # Normalize item areas for color mapping
    norm = plt.Normalize(min([item.shape.area for item in items]), max([item.shape.area for item in items]))

    # Plot the packed items inside the knapsack with colors based on their normalized area
    for item in items:
        normalized_area = norm(item.shape.area)
        color = cmap(normalized_area)  # Get color from the adjusted colormap
        ax.fill(*item.shape.exterior.xy, color=color, edgecolor=cmap(normalized_area/2))

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Set aspect ratio to 'equal'
    ax.set_aspect('equal', 'box')

    # Calculate and display the filling rate at the bottom of the figure
    filling_rate = (sum(item.shape.area for item in items) / knapsack.shape.area) * 100
    filling_rate_text = f'Filling Rate: {round(filling_rate)}%'
    fig.text(0.5, 0.05, filling_rate_text, ha='center', va='bottom', fontsize=10, color='black')

    # Show the plot
    plt.show()


def visualize_knapsacks_and_items(knapsacks_with_items, titles):
    num_knapsacks = len(knapsacks_with_items)
    cols = min(4, num_knapsacks)  # Ensure a maximum of 5 columns
    rows = (num_knapsacks + cols - 1) // cols  # Calculate required rows
    
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
    
    for i, ((knapsack, items), title) in enumerate(zip(knapsacks_with_items, titles)):
        row, col = divmod(i, cols)
        ax = axs[row, col]

        # Set the title for each subplot
        ax.set_title(title, fontsize=15, pad=5)  # pad=3 for a little space between the title and the plot
        
        ax.plot(*knapsack.shape.exterior.xy, color='black', linewidth=1)
        
        original_cmap = plt.cm.viridis
        colors = original_cmap(np.linspace(0.33, 0.66, 256))
        cmap = mcolors.LinearSegmentedColormap.from_list("viridis_middle", colors)
        norm = plt.Normalize(min([item.shape.area for item in items]), max([item.shape.area for item in items]))
        
        for item in items:
            normalized_area = norm(item.shape.area)
            ax.fill(*item.shape.exterior.xy, color=cmap(normalized_area), edgecolor=cmap(normalized_area/2))
        
        # Set axes spines to white to effectively hide them
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        # Hide tick marks and labels
        ax.tick_params(colors='white', which='both')
        
        # Set the aspect ratio
        ax.set_aspect('equal', 'box')
        
        # Set x-axis label with visible color
        filling_rate = (sum(item.shape.area for item in items) / knapsack.shape.area) * 100
        ax.set_xlabel(f'Zapolnjenost: {round(filling_rate)}%', fontsize=15, color='black')
        
        # Ensure the label is positioned correctly and visible
        ax.xaxis.label.set_visible(True)

    # Remove unused subplots
    for j in range(i+1, rows*cols):
        axs.flatten()[j].set_visible(False)
        
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)  # Adjust padding and spacing as needed
    return plt

