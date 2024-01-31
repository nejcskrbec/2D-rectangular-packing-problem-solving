import concurrent.futures
import numpy as np

from shared.shape_utils import *
from shared.knapsack import *
from shared.item import *
from math import sqrt

class Angle:
    def __init__(self, position):
        self.position = position

    def __eq__(self, other):
        if isinstance(other, Angle):
            return (self.position == other.position)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.position)

class Placement:
    def __init__(self, item, position, occupied_angles):
        self.item = item
        self.position = position
        self.occupied_angles = occupied_angles

    def distance_from_origin(self):
        return sqrt(self.position[0]**2 + self.position[1]**2)


def get_shared_vertices(first_item, second_item):
    """Function that returnes vertices that lie on an edge of one shape or vice-versa."""

    # Get vertices for both items
    first_item_vertices = np.array(first_item.get_vertices())
    second_item_vertices = np.array(second_item.get_vertices())

    # Generate line segments for each shape
    first_item_lines = np.array(list(zip(first_item_vertices, np.roll(first_item_vertices, -1, axis=0))))
    second_item_lines = np.array(list(zip(second_item_vertices, np.roll(second_item_vertices, -1, axis=0))))

    # Check for shared vertices
    shared_first = first_item_vertices[is_point_on_line_vec(first_item_vertices, second_item_lines[:, 0], second_item_lines[:, 1])]
    shared_second = second_item_vertices[is_point_on_line_vec(second_item_vertices, first_item_lines[:, 0], first_item_lines[:, 1])]

    # Combine and get unique vertices
    shared_vertices = np.unique(np.concatenate((shared_first, shared_second), axis=0), axis=0)

    return shared_vertices.tolist()


def get_common_vertices(first_item, second_item):
    """Function that returns the vertices which are in both items"""

    # Convert vertices to structured arrays
    dtype = [('x', float), ('y', float)]
    first_vertices = np.array(first_item.get_vertices(), dtype=dtype)
    second_vertices = np.array(second_item.get_vertices(), dtype=dtype)

    # Find common vertices
    shared_vertices_structured = np.intersect1d(first_vertices, second_vertices)

    # Convert structured array back to list of tuples and return
    return [tuple(vertex) for vertex in shared_vertices_structured]


def get_all_item_angles(item, knapsack):
    """Function that calculates all angles for an item in the knapsack, based on the angle definition."""

    # Get bounds of the knapsack
    minx, miny, maxx, maxy = knapsack.get_bounds()

    # Get vertices of the item
    item_vertices = np.array(item.get_vertices())

    # Create an array of all projections for x and y coordinates separately
    projections_x = np.array([(vertex[0], miny) for vertex in item_vertices] + 
                            [(vertex[0], maxy) for vertex in item_vertices])
    projections_y = np.array([(minx, vertex[1]) for vertex in item_vertices] + 
                            [(maxx, vertex[1]) for vertex in item_vertices])

    # Collect all points
    all_points = np.concatenate((item_vertices, projections_x, projections_y))

    # Create Angle objects, checking if each point is in shared vertices
    angles = [Angle(tuple(point)) 
            for point in all_points]

    # Remove duplicate angles by converting to a set and back to a list
    unique_angles = list(set(angles))

    return unique_angles


def get_unique_angles_from_items(knapsack):
    """Function that generates all angles for the current items in the knapsack"""

    # Extract angle positions as tuples
    angle_positions = [tuple(angle.position) for item in knapsack.items for angle in get_all_item_angles(item, knapsack)]

    # Convert to a NumPy array
    angles_array = np.array(angle_positions)

    # Get unique angles
    unique_angle_positions = np.unique(angles_array, axis=0)

    # Convert back to Angle objects
    unique_angles = [Angle(position=tuple(pos)) for pos in unique_angle_positions]

    return unique_angles


def calculate_occupied_angles(knapsack, item, angles):
    """Function that calculates which angles are occupied by an item at a certain position"""

    # Adjust placed item's vertices based on position
    item_vertices = np.array(item.get_vertices())
    
    # Generate line segments for the placed item
    item_lines = np.array(list(zip(item_vertices, np.roll(item_vertices, -1, axis=0))))

    # Convert the list of unique angles to a NumPy array for their positions
    unique_angles_array = np.array([angle.position for angle in angles])

    # Check if angles lie on placed item's edges
    covered_by_edges = is_point_on_line_vec(unique_angles_array, item_lines[:, 0], item_lines[:, 1])

    # Check if angles coincide with any of the item's vertices
    covered_by_vertices = np.any(np.all(unique_angles_array[:, None] == item_vertices, axis=-1), axis=-1)

    # Get the Angle objects of the total covered angles
    covered_angles = [angles[i] for i in np.where(np.logical_or(covered_by_edges, covered_by_vertices))[0]]

    return covered_angles


def generate_placements_for_item(knapsack, item):
    """Function that generates all possible placements for an item inside of a knapsack."""

    # Pre-compute angles for all items
    all_angles = {current_item: get_all_item_angles(current_item, knapsack) for current_item in knapsack.items}

    # Create a set of unique angles from all items
    unique_angles = list(set(angle for angles in all_angles.values() for angle in angles))

    # Generate placements using list comprehension with unique angles
    placements = [
        Placement(item, angle.position, calculate_occupied_angles(knapsack, item, unique_angles))
        for angle in unique_angles
        if knapsack.is_valid_placement(item, angle.position, 'left')
    ]

    return placements


def generate_candidate_aops(knapsack, items):
    """Function that generates candidate aops for a list of items which have not yet beed placed inside of the knapsack."""

    # Generate initial candidate placements for all items
    candidate_aops = [placement for item in items.copy() for placement in generate_placements_for_item(knapsack, item)]

    # Sort candidate placements
    candidate_aops.sort(key=lambda p: (-len(p.occupied_angles), p.distance_from_origin()))

    return candidate_aops


def local_best_fit_first(knapsack, items):
    knapsack_copy = copy.deepcopy(knapsack)
    items_copy = copy.deepcopy(items)

    if len(knapsack_copy.items) == 0:
        item = items_copy[0]
        items_copy.remove(item)
        knapsack_copy.add_item(item, (0,0), 'left')

    aop_max = None
    items_copy = sorted(items_copy, key=lambda item: item.shape.area)

    # Generate initial candidate placements for all items
    candidate_aops = generate_candidate_aops(knapsack_copy, items_copy)

    # Iterate until all candidate placements are processed
    while candidate_aops and items_copy:
        # Select the best placement
        aop_max = max(candidate_aops, key=lambda p: (len(p.occupied_angles), -p.distance_from_origin()))

        # Add item to knapsack
        knapsack_copy.add_item(aop_max.item, aop_max.position, 'left')

        # Remove the placed item from the list of items
        items_copy.remove(aop_max.item)

        candidate_aops = generate_candidate_aops(knapsack_copy, items_copy)

    # Return the items in the knapsack and the area occupied
    return knapsack_copy.items, knapsack_copy.get_area()


def BFHA_local(knapsack, items):
    knapsack_copy = copy.deepcopy(knapsack)
    items_copy = copy.deepcopy(items)

    M = sorted(items_copy, key=lambda item: item.shape.area, reverse=True)
    max_filling_rate = 0
    max_layout = []

    M_copy = copy.deepcopy(M)

    for item in M:
        M.remove(item)

        # Generate initial layout
        items_copy.add_item(item, (0,0), 'left')

        final_layout, filling_rate = local_best_fit_first(items_copy, M)

        if filling_rate > max_filling_rate:
            max_filling_rate = filling_rate
            max_layout = final_layout
        
        items_copy.items = []
        M = copy.deepcopy(M_copy)

    return max_layout, max_filling_rate


def local_best_fit_first_parallel(knapsack, items):
    # This is a wrapper function for local_best_fit_first to be used with multiprocessing.
    # It needs to be at the top level (not nested) for multiprocessing to work properly.
    return local_best_fit_first(knapsack, items)


def BFHA_local_optimized(knapsack, items):
    knapsack_copy = copy.deepcopy(knapsack)
    items_copy = copy.deepcopy(items)

    M = sorted(items_copy, key=lambda item: item.shape.area, reverse=True)
    max_filling_rate = 0
    max_layout = []
    M_copy = copy.deepcopy(M)

    # Create a process pool executor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for item in M:
            M.remove(item)
            knapsack_copy.add_item(item, (0,0), 'left')

            # Submit the local_best_fit_first function to the executor
            future = executor.submit(local_best_fit_first_parallel, copy.deepcopy(knapsack_copy), M)
            futures.append(future)

            knapsack_copy.items = []
            M = copy.deepcopy(M_copy)

        # Collect the results as they are completed
        for future in concurrent.futures.as_completed(futures):
            final_layout, filling_rate = future.result()

            if filling_rate > max_filling_rate:
                max_filling_rate = filling_rate
                max_layout = final_layout

    return max_layout, max_filling_rate
        

def global_best_fit_first(knapsack, items, aop):
    knapsack_copy = copy.deepcopy(knapsack)
    items_copy = copy.deepcopy(items)
    knapsack_copy.add_item(aop.item, aop.position, 'left')
    items_copy.remove(aop.item)
    final_layout, filling_rate = local_best_fit_first(knapsack_copy, items_copy)
    return filling_rate


def BFHA(knapsack, items):
    M = sorted(items, key=lambda item: item.shape.area, reverse=True)
    L = []
    max_filling_rate = 0
    max_layout = []

    M_copy = copy.deepcopy(M)

    for item in M:
        M.remove(item)

        # Generate initial layout
        knapsack.add_item(item, (0,0), 'left')

        # Generate candidate aops
        L = generate_candidate_aops(knapsack, M)

        while L:
            max_benefit = 0
            aop_max = None
            for aop in L:
                benefit = global_best_fit_first(knapsack, M, aop)
                if aop_max is not None:
                    if benefit > max_benefit:
                        max_benefit = benefit
                        aop_max = aop
                else:
                    aop_max = aop
                
            # Add item to knapsack
            knapsack.add_item(aop_max.item, aop_max.position, 'left')

            # Remove the placed item from the list of items
            M.remove(aop_max.item)

            # Regenerate candidate placements for remaining items
            L = generate_candidate_aops(knapsack, M)

        filling_rate = knapsack.get_area()
        if filling_rate > max_filling_rate:
            max_filling_rate = filling_rate
            max_layout = knapsack.items
        
        knapsack.items = []
        M = copy.deepcopy(M_copy)
        L = []

    return max_layout, max_filling_rate