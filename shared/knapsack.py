import numpy as np

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from itertools import *

from shared.shape_utils import *
from shared.shape import *
from shared.item import *

class Knapsack(Shape):
    
    def __init__(self, width, height, position=None, rotation=None):
        super().__init__(width, height, position, rotation)
        self.angles = self.shape.bounds
        self.items = []

    def remove_item(self, item):
        self.items.remove(item)

    
    def add_item(self, item, position, mode):
        if item not in self.items:
            if self.is_valid_placement(item, position, mode):
                self.items.append(item)
                return True
        return False
    

    def is_valid_placement(self, item, position, mode):
        if mode == 'bottom_left':
            item.place_bottom_left(position)
        elif mode == 'top_left':
            item.place_top_left(position)
        elif mode == 'bottom_right':
            item.place_bottom_right(position)
        elif mode == 'top_right':
            item.place_top_right(position)

        if not Shape.does_shape_contain_other(self.shape, item.shape):
            return False

        items_combined_shape = self.get_items_combined_shape()
        polygons = [p for p in ([items_combined_shape]
                    if not isinstance(items_combined_shape, MultiPolygon) 
                    else list(items_combined_shape.geoms)) 
                    if not p.is_empty]
    
        if any(Shape.do_shapes_overlap(item.shape, polygon)
            for polygon in polygons):
            return False

        return True


    def get_area(self):
        return sum(item.shape.area for item in self.items)


    def get_items_combined_shape(self):
        shapes = [x.shape for x in self.items]
        return unary_union(shapes)


    def get_items_combined_shape_vertices(self):
        items_combined_shape = self.get_items_combined_shape()
        polygons = [p for p in ([items_combined_shape]
                    if not isinstance(items_combined_shape, MultiPolygon) 
                    else list(items_combined_shape.geoms)) 
                    if not p.is_empty]

        # Extract unique vertices using list comprehension
        unique_vertices = set(
            point 
            for poly in polygons 
            for geometry in ([poly] if isinstance(poly, Polygon) else poly.geoms)
            for point in (list(geometry.exterior.coords) + [pt for interior in geometry.interiors for pt in interior.coords])
        )

        return list(unique_vertices)