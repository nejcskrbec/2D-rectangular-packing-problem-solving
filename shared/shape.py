import copy as cp

from shapely.affinity import translate, rotate
from shapely.geometry import Polygon
from itertools import combinations as combs

class Shape:

    def __init__(self, width, height, position=None, rotation=None):
        self.shape = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if position:
            self.translate(position)
        if rotation:
            self.rotate(rotation)
        self.width = width
        self.height = height


    def translate(self, position):
        dx = position[0] - self.shape.centroid.x
        dy = position[1] - self.shape.centroid.y
        translated_shape = translate(self.shape, xoff=dx, yoff=dy)
        self.shape = translated_shape


    def place_bottom_left(self, point):
        minx, miny, maxx, maxy = self.get_bounds()
        self.translate((point[0] - minx + self.get_centroid().x, point[1] - miny + self.get_centroid().y))


    def place_top_left(self, point):
        minx, miny, maxx, maxy = self.get_bounds()
        self.translate((point[0] - minx + self.get_centroid().x, point[1] + miny - self.get_centroid().y))


    def place_bottom_right(self, point):
        minx, miny, maxx, maxy = self.get_bounds()
        self.translate((point[0] - maxx + self.get_centroid().x, point[1] - miny + self.get_centroid().y))


    def place_top_right(self, point):
        minx, miny, maxx, maxy = self.get_bounds()
        self.translate((point[0] - maxx + self.get_centroid().x, point[1] + miny - self.get_centroid().y))


    def rotate(self, angle_degrees, origin='centroid'):
        rotation_point = self.shape.centroid
        rotated_shape = rotate(self.shape, angle_degrees, origin=rotation_point)
        self.shape = rotated_shape


    def get_bounds(self):
        return self.shape.bounds
    

    def get_vertices(self):
        x, y = self.shape.exterior.coords.xy
        return list(zip(x, y))[:-1]


    def get_centroid(self):
        return self.shape.centroid


    def get_rotation(self):
        return self.shape.rotation


    def get_shape_exterior_points(self):
        return self.shape.exterior.xy
    

    def intersects(self, other):
        return self.shape.intersects(other.shape)


    def copy(self):
        return cp.deepcopy(self)

    @staticmethod
    def do_shapes_overlap(first_shape, second_shape):
        if ((first_shape.intersects(second_shape) == False) and (first_shape.disjoint(second_shape) == True)) or ((first_shape.intersects(second_shape) == True) and (first_shape.touches(second_shape) == True)):
            return False
        elif (first_shape.intersects(second_shape) == True) and (first_shape.disjoint(second_shape) == False) and (first_shape.touches(second_shape) == False):
            return True

    @staticmethod
    def do_any_shapes_overlap(shapes):
        return any(Shape.do_shapes_overlap(shape1, shape2) 
                for shape1, shape2 in combs(shapes, 2))


    @staticmethod
    def does_shape_contain_other(container_shape, content_shape):
        return content_shape.within(container_shape)