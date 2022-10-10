from turtle import right
import numpy as np
from classes import *
from reference_frames import *
from astroConstants import *

''' kep2car:
Conversion function from keplerian to cartesian elements.
Input:
- keplerian_elements : KeplerianElements object
Output:
- position : Cartesian object
- velocity : Cartesian object
'''
def kep2car(keplerian_elements): 
    
    theta = 2*np.arctan(np.sqrt((1 + keplerian_elements.e)/(1 - keplerian_elements.e))*np.tan(keplerian_elements.E/2))
    semi_latus_rectum = keplerian_elements.a * (1 - keplerian_elements.e**2)
    position = semi_latus_rectum / (1 + keplerian_elements.e*np.cos(theta))

    [position_pf, velocity_pf] = perifocalFrame(keplerian_elements)
    R = pf2ge(keplerian_elements) # rotation matrix

    position_ge = R @ position_pf
    velocity_ge = R @ velocity_pf

    position = Cartesian(position_ge[0], position_ge[1], position_ge[2])
    velocity = Cartesian(velocity_ge[0], velocity_ge[1], velocity_ge[2])

    return position, velocity


''' car2kep:
Conversion function from cartesian to keplerian elements.
Input:
- position : Cartesian object
- velocity : Cartesian object
Output:
- keplerian_elements : KeplerianElements object
'''
def car2kep(position, velocity): 
    
    gravitational_parameter = astroConstants(13)
    angular_momentum_vector = np.cross(position.vector(), velocity.vector())
    angular_momentum_norm = np.linalg.norm(angular_momentum_vector)

    energy = 0.5 * (velocity.normalise()**2) - gravitational_parameter/position.normalise()
    
    a = - gravitational_parameter/(2*energy)
    
    e_vector = np.cross(velocity.vector(), angular_momentum_vector)/gravitational_parameter - position.vector()/position.normalise()
    e = np.linalg.norm(e_vector)

    i = np.arccos(angular_momentum_vector[2]/angular_momentum_norm)

    n_vector = np.cross([0, 0, 1], angular_momentum_vector)
    n = np.linalg.norm(n_vector)

    if n_vector[1] >= 0:
        raan = np.arccos(n_vector[0]/n)
    else:
        raan = 2*np.pi - np.arccos(n_vector[0]/n)

    if e_vector[2] >= 0:
        omega = np.arccos(np.dot(n_vector, e_vector)/(n*e))
    else:
        omega = 2*np.pi - np.arccos(np.dot(n_vector, e_vector)/(n*e))

    radial_velocity = np.dot(position.unitVector(), velocity.vector())

    if radial_velocity >= 0:
        theta = np.arccos(np.dot(e_vector, position.vector())/(e*position.normalise()))
    else:
        theta = 2*np.pi - np.arccos(np.dot(e_vector, position.vector())/(e*position.normalise()))

    E = 2*np.arctan(np.sqrt((1 - e)/(1 + e))*np.tan(theta/2))

    keplerian_elements = KeplerianElements(a, e, i, raan, omega, E)

    return keplerian_elements


def position2ra_dec(position):

    # Direction cosines of r
    l = position.x/position.normalise()
    m = position.y/position.normalise()
    n = position.z/position.normalise()

    declination = np.arcsin(n)

    if (m > 0):
        right_ascension = np.arccos(l/np.cos(declination))
    else:
        right_ascension = 2*np.pi - np.arccos(l/np.cos(declination))

    right_ascension = 180/np.pi * right_ascension
    declination = 180/np.pi * declination

    if (right_ascension > 180):
        right_ascension = right_ascension - 360

    return declination, right_ascension



''' kep2eq
Conversion of the keplerian elements into Equinoctial orbital elements
'''
def kep2eq(keplerian_elements):

    a = keplerian_elements.a
    h = keplerian_elements.e * np.sin(keplerian_elements.raan + keplerian_elements.omega)
    k = keplerian_elements.e * np.cos(keplerian_elements.raan + keplerian_elements.omega)
    p = np.tan(keplerian_elements.i/2) * np.sin(keplerian_elements.raan)
    q = np.tan(keplerian_elements.i/2) * np.cos(keplerian_elements.raan)
    F = rad02pi(keplerian_elements.raan + keplerian_elements.omega + keplerian_elements.E)
    equinoctial_elements = EquinoctialElements(a, h, k, p, q, F)
    return equinoctial_elements


def eq2kep(equinoctial_elements):

    a = equinoctial_elements.a
    e = np.sqrt(equinoctial_elements.h**2 + equinoctial_elements.k**2)
    i = 2 * np.arctan( np.sqrt(equinoctial_elements.p**2 + equinoctial_elements.q**2) )
    raan = rad02pi(np.arctan2(equinoctial_elements.p, equinoctial_elements.q))
    omega = rad02pi(np.arctan2(equinoctial_elements.h, equinoctial_elements.k) - raan)
    E = rad02pi(equinoctial_elements.F - raan - omega)
    keplerian_elements = KeplerianElements(a, e, i, raan, omega, E)
    return keplerian_elements


def eq2car(equinoctial_elements):

    keplerian_elements = eq2kep(equinoctial_elements)
    [position, velocity] = kep2car(keplerian_elements)
    return position, velocity


''' rad02pi
Function that make the angle stays in the boundaries 0 and 2pi (2pi excluded)
'''
def rad02pi(angle):
    angle = np.array(angle)
    f = np.floor(abs(angle)/(2*np.pi))

    if (angle.size > 1):
        for j in range(angle.size):
            if (angle[j] < 0):
                angle[j] = angle[j] + 2*np.pi * f[j]
            if (angle[j] >= 2*np.pi):
                angle[j] = angle[j] - 2*np.pi * f[j]
    else:
        if (angle < 0):
            angle = angle + 2*np.pi * f
        if (angle >= 2*np.pi):
            angle = angle - 2*np.pi * f

    return angle


def radpi2pi(angle):
    angle = rad02pi(angle)

    if (angle.size > 1):
        for j in range(angle.size):
            if (angle[j] > np.pi):
                angle[j] = angle[j] - 2*np.pi
    else:
        if (angle > np.pi):
            angle = angle - 2*np.pi

    return angle
