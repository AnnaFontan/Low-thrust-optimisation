import numpy as np
import seaborn as sns

import matplotlib.pyplot as plot
# from scipy.optimize import minimize
from scipy.optimize import Bounds

sns.set_theme(style = "darkgrid")

from classes import *
from conversions import *
from astroConstants import *
from odeOrbits import *
from orbitalPlots import *
from satellite_data import *
from optimisation import *

if __name__ == '__main__':

    gravitational_parameter = astroConstants(13);  # [km***3/s**2] Earth planetary constant
    radius_earth = astroConstants(23)

    # Integration of the ordinary differential equation
    # [year0, month0, day0, hrs0, min0, sec0] = fraction2month_day(2022, 263.27039333)
    number_weights = satellite_data()[7] # equinocital orbital elements of the final (desired) orbit
    amplitude_weights = satellite_data()[10]
    equinoctial_elements_initial = satellite_data()[0]
    equinoctial_elements_final = satellite_data()[6]

    # Minimisation optimisation
    plot_boolean = 0.0 # 0 = no plot, 1 = yes plot
    LB = np.concatenate((np.linspace(0.0, 0.0, 2*number_weights), [plot_boolean]), axis = 0)
    UB = np.concatenate((np.linspace(1.0, 1.0, 2*number_weights), [plot_boolean]), axis = 0)
    BOUNDS = Bounds(LB, UB)
    
    toll = 1
    n_iterations = 4
    n_particles = 2

    [minimo, res] = particleSwarm(minFunction, LB, UB, toll, n_iterations, n_particles)
    res[res.size-1] = 1

    tf_min = minFunction(res)
    print(tf_min)
    '''
    res = minimize(minFunction, X0, args = (y0, 0), method = 'L-BFGS-B', jac = None, bounds = BOUNDS, tol = None, callback = None, options = {'disp': None, 
        'maxcor': 10, 'ftol': 2.0, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
    '''
    weights = np.dot(res[0 : 2*number_weights], amplitude_weights)
    a_vector = np.linspace(equinoctial_elements_initial.a, equinoctial_elements_final.a, number_weights) # [km]

    weights_point1 = - np.sort(- weights[0 : number_weights])
    weights_point2 = - np.sort(- weights[number_weights : 2*number_weights])

    # Figure of the weights
    plot.plot(a_vector, weights_point1)
    plot.scatter(a_vector, weights_point1, label = 'weights a')
    plot.plot(a_vector, weights_point2)
    plot.scatter(a_vector, weights_point2, label = 'weights i')
    plot.axhline(y = 1, linestyle = '-.', label = 'weights e')
    plot.xlabel('a [km]')
    plot.ylabel('Weights')
    plot.tight_layout()
    plot.legend()
    plot.show()