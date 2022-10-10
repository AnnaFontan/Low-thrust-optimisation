"""
class definition : ThisIsPascalCase
class methods : thisIsCamelCase
variable names : this_is_train_case
"""

import numpy as np

class Cartesian:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def vector(self):
        return [self.x, self.y, self.z]

    def normalise(self):
        return np.linalg.norm([self.x, self.y, self.z])

    def unitVector(self):
        return [self.x, self.y, self.z] / self.normalise()


class KeplerianElements:
    def __init__(self, a, e, i, raan, omega, E):
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.omega = omega
        self.E = E

class EquinoctialElements:
    def __init__(self, a, h, k, p, q, F):
        self.a = a
        self.h = h
        self.k = k
        self.p = p
        self.q = q
        self.F = F
