import numpy as np

from shapely.geometry import Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString
from collections import Counter

from shared.shape_utils import *
from shared.knapsack import *
from shared.shape import *
from shared.item import *

def common_perimeter_length(placement, knapsack):
    # Position the item based on its placement type
    if placement.type == 'bottom_left':
        placement.item.place_bottom_left(placement.position)
    elif placement.type == 'top_left':
        placement.item.place_top_left(placement.position)
    elif placement.type == 'bottom_right':
        placement.item.place_bottom_right(placement.position)
    elif placement.type == 'top_right':
        placement.item.place_top_right(placement.position)

    # Get the boundary of the item's shape
    item_boundary = placement.item.shape.boundary

    # Prepare a list of polygons (boundaries) to check against
    items_combined_shape = knapsack.get_items_combined_shape().boundary
    polygons_boundaries = ([items_combined_shape] if not isinstance(items_combined_shape, MultiPolygon) 
                        else [poly.boundary for poly in items_combined_shape.geoms])
    polygons_boundaries.append(knapsack.shape.boundary)

    # Initialize the result
    result = 0

    # Check the item boundary against each polygon's boundary in the knapsack
    for polygon_boundary in polygons_boundaries:
        # Calculate the intersection between item boundary and polygon boundary
        common_boundary = item_boundary.intersection(polygon_boundary)

        # If the common boundary is a LineString or MultiLineString, add its length to the result
        if isinstance(common_boundary, (LineString, MultiLineString)):
            result += common_boundary.length

    return result


class Placement:
    def __init__(self, item, position, knapsack, placement_type):
        self.item = item  
        self.position = position 
        self.type = placement_type
        self.common_perimeter = common_perimeter_length(self, knapsack)
    
    def __eq__(self, other):
        if isinstance(other, Placement):  
            return (
                self.position == other.position and
                self.item == other.item and
                self.type == other.type 
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.position, self.item, self.type))


def get_placements_with_touching_perimiters(knapsack, item):
    current_packing_vertices = list(set(knapsack.get_items_combined_shape_vertices() + knapsack.get_vertices()))

    placements_bottom_left = [Placement(item, position, knapsack, 'bottom_left') for position in current_packing_vertices 
                            if knapsack.is_valid_placement(item, position, 'bottom_left')]
    placements_top_left = [Placement(item, position, knapsack, 'top_left') for position in current_packing_vertices 
                            if knapsack.is_valid_placement(item, position, 'top_left')]
    placements_bottom_right = [Placement(item, position, knapsack, 'bottom_right') for position in current_packing_vertices 
                            if knapsack.is_valid_placement(item, position, 'bottom_right')]
    placements_top_right = [Placement(item, position, knapsack, 'top_right') for position in current_packing_vertices 
                            if knapsack.is_valid_placement(item, position, 'top_right')]

    return placements_bottom_left + placements_top_left + placements_bottom_right + placements_top_right


def touching_perimiter_heuristic(knapsack, items):
    knapsack_copy = cp.deepcopy(knapsack)
    items_copy = cp.deepcopy(items)

    unpackable_items = set()

    while len(items_copy) > 0 and not set(items_copy).issubset(unpackable_items):
        item = items_copy[0]
        all_valid_placements = get_placements_with_touching_perimiters(knapsack_copy, item)
        
        if not all_valid_placements:
            unpackable_items.add(item)
            items_copy.remove(item)
            continue  

        max_placement = max(all_valid_placements, key=lambda p: p.common_perimeter)
        knapsack_copy.add_item(max_placement.item, max_placement.position, max_placement.type)
        items_copy.remove(item)

    return knapsack_copy.items, knapsack_copy.get_area() / (knapsack_copy.width * knapsack_copy.height)
