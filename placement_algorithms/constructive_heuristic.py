import numpy as np

from shapely.geometry import Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString

from shared.shape_utils import *
from shared.knapsack import *
from shared.shape import *
from shared.item import *
from itertools import groupby

class Space(Shape):
    def __init__(self, position, width, height, left_wall_point, right_wall_point):
        super().__init__(width, height, None, None)
        self.left_wall_point = left_wall_point
        self.right_wall_point = right_wall_point
        self.position = position
        self.left = self.left_wall_point[1] - self.position[1]
        self.right = self.right_wall_point[1] - self.position[1]
        
    def __eq__(self, other):
        if isinstance(other, Space):
            return (
                self.position == other.position and
                self.width == other.width and
                self.height == other.height and
                self.left_wall_point == other.left_wall_point and
                self.right_wall_point == other.right_wall_point
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.position)


def construct_empty_space(knapsack, position):
    minx, miny, maxx, maxy = knapsack.get_bounds()

    # Return None if the position is on the top or right edge of knapsack
    if position[0] == maxx or position[1] == maxy:
        return None

    items_combined_shape_vertices = knapsack.get_items_combined_shape_vertices()

    right_projection_points = project_right(position, knapsack)

    # Assuming projection_points_have_item_above_dict is already created as shown previously
    projection_points_have_item_above_dict = {
        point: any(point[0] == vertex[0] and vertex[1] > point[1] for vertex in items_combined_shape_vertices)
        for point in right_projection_points
    }

    # Use the dictionary to determine if all points do not have items above them
    all_have_no_items_above = not any(projection_points_have_item_above_dict.values())

    if all_have_no_items_above:
        width_point = max(projection_points_have_item_above_dict.keys(), 
                        key=lambda point: abs(point[0] - position[0]))
    else:
        width_point = min((point for point, has_above in projection_points_have_item_above_dict.items() if has_above),
                         key=lambda point: abs(point[0] - position[0]), 
                         default=None)

    # Using list comprehension to construct points_above
    points_above = {point: project_up(point, knapsack) for point in right_projection_points}

    # Find the right wall point
    right_wall_candidates = [point for point in points_above[width_point] if point in items_combined_shape_vertices]
    right_wall_point = min(right_wall_candidates, key=lambda p: abs(p[1] - maxy), default=points_above[width_point][-1])

    # Find the left wall point
    left_wall_candidates = [point for point in project_up(position, knapsack) if point in items_combined_shape_vertices]
    left_wall_point = min(left_wall_candidates, key=lambda p: abs(p[1] - maxy), default=project_up(position, knapsack)[-1])
        
    result = Space(position, width_point[0] - position[0], maxy - position[1], left_wall_point, right_wall_point)
    result.place_bottom_left(position)

    return result


def vertical_line_through_point_intersects_shape(knapsack, point, item):
    """Function that checks if a vertical line going through a point intersects with the item"""
    points = [point] + project_up(point, knapsack)
    vertices = item.get_vertices()  # Assuming this returns [bottom_left, bottom_right, top_right, top_left]

    # Opposite vertices pairs
    opposite_vertices = [(vertices[0], vertices[2]), (vertices[1], vertices[3]), 
                         (vertices[2], vertices[0]), (vertices[3], vertices[1])]

    # Check if any consecutive points are on opposite vertices
    if any((point1, point2) in opposite_vertices for point1, point2 in zip(points, points[1:])):
        return True

    # Define edges
    edges = {'left': [vertices[0], vertices[3]], 'bottom_right': [vertices[1], vertices[2]], 
             'top': [vertices[2], vertices[3]], 'bottom': [vertices[0], vertices[1]]}

    # Opposite edges pairs
    opposite_edges = [('left', 'bottom_right'), ('top', 'bottom')]

    # Check if any consecutive points are on opposite edges
    return any((point1 in edges[edge1] and point2 in edges[edge2]) 
               for point1, point2 in zip(points, points[1:]) 
               for edge1, edge2 in opposite_edges)


def is_valid_empty_space(knapsack, empty_space):
    if empty_space:
        return knapsack.is_valid_placement(Item(empty_space.width, empty_space.height), empty_space.position, 'bottom_left')
    else: 
        return False


def update_empty_spaces(knapsack, empty_spaces):
    """Function that accepts the current list of empty spaces and updates it based on the current placement of items in the knapsack."""

    # Filter out none spaces
    empty_spaces = [space for space in empty_spaces if space is not None ]

    ## FIRST STEP: Generate new empty spaces
    minx, miny, maxx, maxy = knapsack.get_bounds()

    # If we have an empty knapsack
    if not knapsack.items or len(knapsack.items) == 0:
        return [Space((0,0), knapsack.width, knapsack.height, (minx, maxy), (maxx, maxy))]

    item = knapsack.items[-1]
    vertices = item.get_vertices()
    bottom_right, top_left = vertices[1], vertices[-1]

    # Construct new spaces based on the positions of the last item
    space_1 = construct_empty_space(knapsack, top_left)
    space_2 = construct_empty_space(knapsack, bottom_right)

    # If we have one item in the knapsack
    if len(knapsack.items) == 1:
        return [space for space in [space_1, space_2] if space is not None ]

    # If we have more than one item in the knapsack
    # Filter valid spaces
    valid_spaces = [space for space in empty_spaces 
                    if is_valid_empty_space(knapsack, space) 
                    and not vertical_line_through_point_intersects_shape(knapsack, space.position, item)]

    # Update spaces that overlap with the new item
    updated_spaces = [construct_empty_space(knapsack, space.position) 
                    for space in empty_spaces if space not in valid_spaces
                    and not vertical_line_through_point_intersects_shape(knapsack, space.position, item)]

    # Project the positions of the existing empty spaces leftwards
    projected_positions = [position for space in empty_spaces 
                       for position in project_left(space.position, knapsack) 
                       if position[0] != minx]

    projected_positions_spaces = [space for position in projected_positions 
                              for space in [construct_empty_space(knapsack, position)]
                              if space and is_valid_empty_space(knapsack, space)]

    # Combine updated_spaces, new_spaces, and projected_positions_spaces, ensuring uniqueness and validity
    unique_spaces = {
        space.position: space
        for space in valid_spaces + updated_spaces + [space_1, space_2] + projected_positions_spaces
        if space is not None and is_valid_empty_space(knapsack, space)
    }
    all_spaces = list(unique_spaces.values())

    # If we have no more valid spaces
    if len(all_spaces) == 0:
        return []

    ### IMPORTANT STEP: Filtering spaces with the same x or y coordinates
    # Initialize filtered_spaces to all_spaces
    filtered_spaces = all_spaces[:]

    # Extract x and y coordinates, and areas
    coordinates = np.array([space.position for space in all_spaces])
    areas = np.array([space.shape.area for space in all_spaces])

    # Find indices where x or y coordinates are shared by more than one space
    _, indices_x, counts_x = np.unique(coordinates[:, 0], return_inverse=True, return_counts=True)
    _, indices_y, counts_y = np.unique(coordinates[:, 1], return_inverse=True, return_counts=True)
    shared_indices = np.where((counts_x[indices_x] > 1) | (counts_y[indices_y] > 1))[0]

    # Extract shared coordinates using these indices
    shared_coords = coordinates[shared_indices]

    if len(shared_coords) < 1:
        return filtered_spaces

    # Generate groups based on shared_coords using list comprehension
    overlapping_groups = [
        [all_spaces[i] for i in np.where((coordinates[:, 0] == coord[0]) | (coordinates[:, 1] == coord[1]))[0]]
        for coord in shared_coords
        if Shape.do_any_shapes_overlap([all_spaces[i].shape for i in np.where((coordinates[:, 0] == coord[0]) | (coordinates[:, 1] == coord[1]))[0]])
    ]

    # Find all spaces that are included in some group
    included_spaces = {space for group in overlapping_groups for space in group}

    # Find all spaces not included in any group
    excluded_spaces = [space for space in all_spaces if space not in included_spaces]

    # Add each excluded space as a single-element group to overlapping_groups
    overlapping_groups.extend([[space] for space in excluded_spaces])

    # Step 4: The rest of your filtering code would remain the same
    filtered_spaces = [
        space for space in filtered_spaces
        if not any(space in group and space != group[np.argmax([areas[all_spaces.index(gs)] for gs in group])] 
                for group in overlapping_groups)
    ]

    return sorted([space for space in filtered_spaces if space is not None ], key=lambda space: space.position[0]**2 + space.position[1]**2)
                

def fitness(item, empty_space):
    if empty_space.left >= empty_space.right:
        if empty_space.width == item.width and empty_space.left == item.height:
            return 4
        elif empty_space.width == item.width and empty_space.left < item.height and empty_space.height > item.height:
            return 3
        elif empty_space.width == item.width and empty_space.left > item.height:
            return 2
        elif empty_space.width > item.width and empty_space.left == item.height:
            return 1
        elif empty_space.width > item.width and empty_space.height >= item.height:
            return 0
        else:
            return -1
    else:
        if empty_space.width == item.width and empty_space.right == item.height:
            return 4
        elif empty_space.width == item.width and empty_space.right < item.height and empty_space.height > item.height:
            return 3
        elif empty_space.width == item.width and empty_space.right > item.height:
            return 2
        elif empty_space.width > item.width and empty_space.right == item.height:
            return 1
        elif empty_space.width > item.width and empty_space.height >= item.height:
            return 0
        else:
            return -1


def calculate_fitness(empty_space, items):
    return {item: fitness(item, empty_space) for item in items}

def constructive_heuristic(knapsack, items):
    knapsack_copy = cp.deepcopy(knapsack)
    items_copy = cp.deepcopy(items)

    # Generate initial empty space
    empty_spaces = update_empty_spaces(knapsack_copy, [])
    
    if len(empty_spaces) > 0:
        s = empty_spaces[0]

        while len(empty_spaces) > 0 and len(items_copy) > 0 and empty_spaces[0].position[1] < knapsack_copy.height:
            minimum_width = min(item.width for item in items_copy)
            s = empty_spaces[0]
            if s.width >= minimum_width:
                fitness_scores = calculate_fitness(s, items_copy)
                best_item = max(fitness_scores, key=fitness_scores.get)
                if fitness_scores[best_item] >= 0:
                    if s.left >= s.right:
                        # Pack item at the bottom left vertex of the empty space (near the left wall)
                        packing_position = s.get_vertices()[0] 
                        knapsack_copy.add_item(best_item, packing_position, 'bottom_left')
                        items_copy.remove(best_item)
                        empty_spaces = update_empty_spaces(knapsack_copy, empty_spaces)
                        empty_spaces = sorted(empty_spaces, key=lambda space: distance_between_points(space.position, (0,0)))
                    else:
                        # Pack item at the bottom right vertex of the empty space (near the right wall)
                        packing_position = s.get_vertices()[1] 
                        knapsack_copy.add_item(best_item, packing_position, 'bottom_right')
                        items_copy.remove(best_item)
                        empty_spaces = update_empty_spaces(knapsack_copy, empty_spaces)
                        empty_spaces = sorted(empty_spaces, key=lambda space: distance_between_points(space.position, (0,0)))
                else:
                    empty_spaces.remove(s)
                    empty_spaces = sorted(empty_spaces, key=lambda space: distance_between_points(space.position, (0,0)))
            else:
                empty_spaces.remove(s)
    
    return knapsack_copy.items, knapsack_copy.get_area() / (knapsack_copy.width * knapsack_copy.height)