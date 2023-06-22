import numpy as np
from math import cos, sin, radians


class Coord_3D:
    def __init__(self, x, y, z, w=1):
        self.x = x / w
        self.y = y / w
        self.z = z / w
        self.w = 1.0

    @staticmethod
    def fromArr(arr):
        return Coord_3D(*arr)

    def setArr(self, arr):
        self.x, self.y, self.z, self.w = map(lambda x: float(x), [*arr])

    def asArr(self):
        return np.array([self.x, self.y, self.z, self.w])

    def get3D(self):
        return np.array([self.x, self.y, self.z])

    def translate(self, by):
        mat = self.__getBaseTransformMatrix()
        mat[0:3, 3] = by
        arr = self.asArr()
        self.setArr(mat.dot(arr))

    def scale(self, by):
        mat = self.__getBaseTransformMatrix()
        if type(by) == int or type(by) == float:
            by = np.full((3), by)
        np.fill_diagonal(mat, [*by, 1])
        arr = self.asArr()
        self.setArr(mat.dot(arr))

    def rotZ(self, angle):
        mat = self.__getBaseTransformMatrix()
        angle = radians(angle)
        mat[0:2, 0:2] = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        arr = self.asArr()
        self.setArr(mat.dot(arr))

    def rotX(self, angle):
        mat = self.__getBaseTransformMatrix()
        angle = radians(angle)
        mat[1:3, 1:3] = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        arr = self.asArr()
        self.setArr(mat.dot(arr))

    def rotY(self, angle):
        mat = self.__getBaseTransformMatrix()
        angle = radians(angle)
        mat[0, 0] = cos(angle)
        mat[0, 2] = sin(angle)
        mat[2, 0] = -sin(angle)
        mat[2, 2] = cos(angle)
        arr = self.asArr()
        self.setArr(mat.dot(arr))

    def __getBaseTransformMatrix(self):
        mat = np.zeros((4, 4))
        mat[0, 0] = 1.0
        mat[1, 1] = 1.0
        mat[2, 2] = 1.0
        mat[3, 3] = 1.0
        return mat

    def __str__(self) -> str:
        return f"Coord({self.x}, {self.y}, {self.z}, {self.w})"

    def __repr__(self) -> str:
        return self.__str__()


def testCoords():
    coord = Coord_3D.fromArr([3, 7, 2])
    print(coord)
    coord.translate([5, 2, 2])
    print(coord)
    coord.rotY(90)
    print(coord)
    coord.scale(3)
    print(coord)


if __name__ == "__main__":
    testCoords()
