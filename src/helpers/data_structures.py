from src.helpers.constants import SCREEN


class Point:
    def __init__(self, x, y):
        self.pos = y * SCREEN['WIDTH'] + x

    @property
    def x(self):
        return self.pos % SCREEN['WIDTH']

    @property
    def y(self):
        return self.pos // SCREEN['WIDTH']

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return Point(self.x + getattr(other, 'x', other), self.y + getattr(other, 'y', other))

    def __sub__(self, other):
        return Point(self.x - getattr(other, 'x', other), self.y - getattr(other, 'y', other))

    def __mul__(self, other):
        return Point(self.x * getattr(other, 'x', other), self.y * getattr(other, 'y', other))

    def __truediv__(self, other):
        return Point(self.x / getattr(other, 'x', other), self.y / getattr(other, 'y', other))

    def __floordiv__(self, other):
        return Point(self.x // getattr(other, 'x', other), self.y // getattr(other, 'y', other))

    def __mod__(self, other):
        return Point(self.x % getattr(other, 'x', other), self.y % getattr(other, 'y', other))
