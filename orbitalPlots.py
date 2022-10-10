import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from itertools import chain

from classes import *
from conversions import *


def sphere(centre, radius, n_meridians = 1e3, n_circles_latitude = None):
    
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)

    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]

    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)

    return sphere_x, sphere_y, sphere_z


''' drawOrbit:
Find the points of an orbit given initial and final true anomaly and draw it in the 3D space without any perturbations added (no ODE)
Input:
- keplerian_elements_initial : KeplerianElements object
- keplerian_elements_final : KeplerianElements object
Output:
- position : Cartesian object
- velocity : Cartesian object
'''
def drawOrbit(keplerian_elements, theta_initial, theta_final):

    theta_step = 0.01 # [rad]

    cont = theta_initial
    if cont >= theta_final:
        cont = cont - 2*np.pi

    plot.figure(figsize=(6,5))
    axes = plot.axes(projection='3d')
    print(type(axes))
    axes.set_xlabel('x [km]')
    axes.set_ylabel('y [km]')
    axes.set_zlabel('z [km]')

    # Conversion of each point in the keplerian frame to cartesian
    X = []
    Y = []
    Z = []

    while cont <= theta_final:
        keplerian_elements_cont = KeplerianElements(keplerian_elements.a, keplerian_elements.e, keplerian_elements.i, keplerian_elements.raan, keplerian_elements.omega, cont, keplerian_elements.mu)
        [position_cont, velocity_final] = kep2car(keplerian_elements_cont)

        X.append(position_cont.x)
        Y.append(position_cont.y)
        Z.append(position_cont.z)

        cont = cont + theta_step
    
    axes.plot3D(X, Y, Z)
    plot.show()



def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')