import numpy as np
from astroConstants import *
from classes import *
from astroConstants import *
from conversions import *
from scipy.optimize import fsolve

def satellite_data():

    radius_earth = astroConstants(23)
    g0 = astroConstants(4) # [km/s**2]
    
    mass0 = 450 # [kg]

    a0 = 3.82 * radius_earth # [km]
    e0 = 0.731 # [adim]
    i0 = 27 * np.pi/180 # [rad]
    raan0 = 99 * np.pi/180 # [rad]
    omega0 = 0 * np.pi/180 # [rad]
    mean_anomaly0 = 90 * np.pi/180 # [rad] ?? i chose it
    fun = lambda E : E - e0*np.sin(E) - mean_anomaly0
    E0 = fsolve(fun, mean_anomaly0)[0]

    Isp = 3300 # [s]
    eta = 0.65
    P0 = 5*1e-3 # [kg km**2 / s**3]
    D = 1 # no degradation effect
    c = Isp*g0 # [km/s] exhaust velocity

    keplerian_elements0 = KeplerianElements(a0, e0, i0, raan0, omega0, E0)
    equinoctial_elements0 = kep2eq(keplerian_elements0)

    year0 = 2000; month0 = 1; day0 = 1; hrs0 = 12; min0 = 0; sec0 = 0
    t0 = 24*3600 * date2J2000(year0, month0, day0, hrs0, min0, sec0)  # [s] launch date
    tf_max = t0 + 365*24*3600 # [s] final time maximum

    # Desired final orbit
    keplerian_elements_final = KeplerianElements(42164, 0, 0, 0, 0, 0)
    equinoctial_elements_final = kep2eq(keplerian_elements_final)

    # Number of weights (eccentricity/inclination)
    number_weights = 3
    amplitude_weights = 20

    return equinoctial_elements0, mass0, Isp, eta, P0, t0, equinoctial_elements_final, number_weights, D, tf_max, amplitude_weights
