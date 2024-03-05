import uuid

from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate

from shared.shape import *

class Item(Shape):
    def __init__(self, width, height, position=None, rotation=None):
        super().__init__(width, height, position, rotation)
        self.id = uuid.uuid4()  # Generate a random GUID

    def translate(self, position):
        dx = position[0] - self.shape.centroid.x
        dy = position[1] - self.shape.centroid.y
        translated_shape = translate(self.shape, xoff=dx, yoff=dy)
        self.shape = translated_shape

    def rotate(self, angle_degrees, origin='centroid'):
        rotation_point = self.shape.centroid
        rotated_shape = rotate(self.shape, angle_degrees, origin=rotation_point)
        self.shape = rotated_shape

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)

