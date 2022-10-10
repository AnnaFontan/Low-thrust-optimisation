import numpy as np
from astroConstants import *
from classes import *
from conversions import *
from satellite_data import *
from reference_frames import *
from scipy.interpolate import interp1d
import matplotlib.pyplot as plot
from scipy.optimize import fsolve

def minFunction(input):
    
    t0 = satellite_data()[5] # [s] launch date
    [equinoctial_elements0, mass0] = satellite_data()[0:2]
    y0 = [equinoctial_elements0.a, equinoctial_elements0.h, equinoctial_elements0.k, equinoctial_elements0.p, equinoctial_elements0.q, mass0]
    X = y0 # initial values for the equinoctial orbital elements + mass

    [equinoctial_elements_final, number_weights] = satellite_data()[6:8]
    keplerian_elements_final = eq2kep(equinoctial_elements_final)

    weights_vectors = input[0 : 2*number_weights] # weights (2*number_weights)
    tf_max = satellite_data()[9] # fixed
    plot_boolean = input[2*number_weights] # fixed

    step = 2*24*3600 # [s] two days

    N = round( (tf_max - step - t0)/step )
    time = np.linspace(t0, tf_max, N)  
    
    X = np.zeros((N, 6))
    X[0] = y0
    j = 0
    final_vect = [keplerian_elements_final.a, keplerian_elements_final.e, keplerian_elements_final.i]
    toll = [1e3, 1e-3, 1e-2]
    tf_min = np.ones(3) # vector of the minimum times
    found = 0
    # Huen's method to analytcally integrate the equation of motion
    while (j < N-1) and (found < 3): # for j in range(N-1):
        f_i = odeFunction(time[j], X[j], weights_vectors)
        X_tilde = X[j] + (time[j+1] - time[j])*f_i 
        f_i1 = odeFunction(time[j+1], X_tilde, weights_vectors)

        X[j+1] = X[j] + (time[j+1] - time[j])/2*(f_i + f_i1) # new values
        e_f = np.sqrt((X[j+1,1])**2 + (X[j+1,2])**2)
        i_f = 2*np.arctan(np.sqrt((X[j+1,3])**2 + (X[j+1,4])**2))
        
        if (abs(X[j+1,0] - final_vect[0]) < toll[0]) and (tf_min[0] == 1):
            tf_min[0] = time[j+1] - t0
            found += 1
        
        if (abs(e_f - final_vect[1]) < toll[1]) and (tf_min[1] == 1):
            tf_min[1] = time[j+1] - t0
            found += 1

        if (abs(i_f - final_vect[2]) < toll[2]) and (tf_min[2] == 1):
            tf_min[2] = time[j+1] - t0
            found += 1

        j += 1

    error = max(tf_min)/(3600*24) 
    time_vector = np.linspace(0, error, N)
    
    #print(error)
    
    # Plot
    if (plot_boolean == 1):
        plot.subplot(2, 2, 1)
        plot.plot(time_vector, X[0:N, 0]) # a
        plot.axhline(y = keplerian_elements_final.a, color = 'r', linestyle = '--')
        plot.xlabel('Time [days]')
        plot.xlim([0, error])
        plot.ylabel('a [km]')

        plot.subplot(2, 2, 2)
        plot.plot(time_vector, np.sqrt( X[0:N, 1]**2 + X[0:N, 2]**2) ) # e
        plot.axhline(y = keplerian_elements_final.e, color = 'r', linestyle = '--')
        plot.xlabel('Time [days]')
        plot.xlim([0, error])
        plot.ylabel('e [adim]')

        plot.subplot(2, 2, 3)
        plot.plot(time_vector, 2*np.arctan( np.sqrt( X[0:N, 3]**2 + X[0:N, 4]**2 ) ) ) # i
        plot.axhline(y = keplerian_elements_final.i, color = 'r', linestyle = '--')
        plot.xlabel('Time [days]')
        plot.xlim([0, error])
        plot.ylabel('i [rad]')
        
        plot.subplot(2, 2, 4)
        plot.plot(time_vector, X[0:N, 5]) # mass
        plot.xlabel('Time [days]')
        plot.xlim([0, error])
        plot.ylabel('mass')

        plot.show()

    return error


''' odeFunction
Function to integrate
Input:
    y : [equinoctial_elements (array), mass]
Output: 
    first derivatives of the input vector
'''
def odeFunction(t, X, weights_vectors):

    # [X = a, h, k, p, q, mass]
    number_weights = satellite_data()[7]
    weight_e = 1 # fixed
    [weight_a, weight_i] = weightsLinearDistribution(X[0], weights_vectors[0 : 2*number_weights])
    
    weights = [weight_a, weight_e, weight_i]
    # F_ex = - np.pi # F_ex: F exit from Earth shadow (lower boundary for the integration)
    # F_in = np.pi # F_in: F entrance into Earth shadow (upper boundary for the integration)

    [F_ex, F_en] = shadowEffect(t, X)
    F_steps = np.linspace(-np.pi, np.pi, 40) # F_ex , F_en

    if (F_ex > F_en):
        idx1 = np.where(F_steps <= F_en)
        idx2 = np.where(F_steps >= F_ex)
        idx_tot = np.concatenate((idx1, idx2), axis = 1)
    else:
        idx1 = np.where(F_steps <= F_en)
        idx2 = np.where(F_steps[idx1[0]] >= F_ex)
        idx_tot = idx2[0]

    integral = 0
    for j in range(F_steps.size-1): # trapezoidal rule to evaluate the integral        
        delta_F = F_steps[j+1] - F_steps[j]
        delta_fun = 0
        if np.where(idx_tot == j):
            delta_fun = delta_fun + firstDerivatives(F_steps[j], X, weights, 0)
        else:
            delta_fun = delta_fun + firstDerivatives(F_steps[j], X, weights, 1)
        
        delta_fun = delta_fun + firstDerivatives(F_steps[j+1], X, weights, 1)
        
        integral = integral + delta_fun/2*delta_F
        # the X is not updated at each step: all of the equinoctial elements (+ mass) remaines constant over the period, the F is the only one that changes!
    
    return integral


def weightsLinearDistribution(a, weights):
# Linear interpolation of the weights.

    amplitude_weights = satellite_data()[10]
    equinoctial_elements_initial = satellite_data()[0]
    equinoctial_elements_final = satellite_data()[6]
    a_min = min( [equinoctial_elements_initial.a, equinoctial_elements_final.a] )
    a_max = max( [equinoctial_elements_initial.a, equinoctial_elements_final.a] )

    if (a < a_min):
        a = a_min
    if (a > a_max):
        a = a_max

    number_weights = satellite_data()[7]
    weights_point1 = - np.sort(- weights[0 : number_weights])
    weights_point2 = - np.sort(- weights[number_weights : 2*number_weights])

    a_vector = np.linspace(equinoctial_elements_initial.a, equinoctial_elements_final.a, num = number_weights, endpoint = True)
    weight1 = interp1d(a_vector, weights_point1)
    weight2 = interp1d(a_vector, weights_point2)

    return np.dot(amplitude_weights, [weight1(a), weight2(a)])



def positionSun(t):
    AU = astroConstants(2) # [km]
    
    # According to The Astronomical Almanac (2013):
    JD = t/(3600*24) # [days] Julian date
    n = JD - 2451545 # [days] number of days since J2000

    L = 280.459 + 0.98564736*n # [deg] mean longitude Sun
    while (L < 0):
        L = L + 360
    while (L > 2*np.pi):
        L = L - 360

    M = np.pi/180 * (357.529 + 0.98560023*n) # [rad] mean anomaly Sun

    lambdaSun = np.pi/180 * (L + 1.915*np.sin(M) + 0.02*np.sin(2*M)) # [rad] solar ecliptic longitude
    epsilon = np.pi/180 * (23.439 - 3.56*n*1e-7) # [rad] obliquity
    r_earth2sun = np.array([np.cos(lambdaSun), np.cos(epsilon)*np.sin(lambdaSun), np.sin(epsilon)*np.sin(lambdaSun)]) # geocentric equatorial frame
    # r_earth2sun = np.dot(r_earth2sun, (1.00014 - 0.01671*np.cos(M) - 0.00014*np.cos(2*M)) * AU) # [km] distance Sun-Earth

    return r_earth2sun


def shadowEffect(t, X):

    radius_earth = astroConstants(23)

    a = X[0]
    h = X[1]
    k = X[2]
    G = np.sqrt(1 - k**2 - h**2)
    b = 1/(1 + G)

    # Sun-Earth:
    [f_hat, g_hat, w_hat] = equinoctialFrame(X)
    R = [f_hat, g_hat, w_hat]

    r_earth2sun = positionSun(t)
    r_sun_hat = R @ r_earth2sun
    x_sun = np.dot(r_sun_hat, f_hat)
    y_sun = np.dot(r_sun_hat, g_hat)
    z_sun = np.dot(r_sun_hat, w_hat)

    # Coefficients
    b1 = 1 - h**2*b
    b2 = h*k*b
    b3 = 1 - k**2*b

    d1 = 1 - x_sun**2
    d2 = 1 - y_sun**2
    d3 = 2*x_sun*y_sun

    h1 = d1*(b1**2 - b2**2) + d2*(b**2 - b3**2) - d3*(b1*b2 - b2*b3)
    h2 = -2*d1*k*b1 - 2*d2*h*b2 + d3*(k*b2 + h*b1)
    h3 = d1*(b2**2 + k**2) + d2*(b3**2 + h**2) - d3*(b2*b3 + h*k) - (radius_earth/a)**2
    h4 = 2*b1*b2*d1 + 2*b2*b3 - d3*(b2**2 + b1*b3)
    h5 = -2*k*b2*d1 - 2*h*b3*d2 + d3*(k*b3 + h*b2)

    A0 = h1**2 + h4**2
    A1 = 2*h1*h2 + 2*h4*h5
    A2 = h2**2 + 2*h1*h3 - h4**2 + h5**2
    A3 = 2*h3*h2 - 2*h4*h5
    A4 = h3**2 - h5**2

    sF = lambda F : np.sin(F)
    cF = lambda F : np.cos(F)
    X1 = lambda F : a * ((1 - (h**2)*b)*cF(F) + h*k*b*sF(F) - k)
    Y1 = lambda F : a * ((1 - (k**2)*b)*sF(F) + h*k*b*cF(F) - h)
    dX_dF = lambda F : a*( -(1 - h**2*b)*sF(F) + h*k*b*cF(F) )
    dY_dF = lambda F : a*( -h*k*b*sF(F) + (1 - k**2*b)*cF(F) )
    dS_dF = lambda F : 2*( (1 - x_sun**2)*X1(F) - x_sun*y_sun*Y1(F) )*dX_dF(F) + 2*( (1 - y_sun**2)*Y1(F) - x_sun*y_sun*Y1(F) )*dY_dF(F)
    S = lambda F : A0*(cF(F))**4 + A1*(cF(F))**3 + A2*(cF(F))**2 + A3*cF(F) + A4

    roots = rad02pi( fsolve(S, np.linspace(-np.pi, np.pi, 15)) )
    idx_erase = []
    for j in range(roots.size-1):
        for i in range(j+1, roots.size):
            if (abs(roots[j] - roots[i]) < 1e-1) or (S(roots[i]) > 1e-2):
                idx_erase.append(i)
    roots = np.delete(roots, idx_erase)
    
    signs = np.where(X1(roots)*x_sun + Y1(roots)*y_sun < 0)
    roots_res = roots[signs]

    F_en = radpi2pi( np.dot(dS_dF(roots_res) < 0, roots_res) )
    F_ex = radpi2pi( np.dot(dS_dF(roots_res) > 0, roots_res) )

    # S = 0
    # entry angle dS_dF < 0
    # exit angle dS_dF > 0

    return F_ex, F_en


def firstDerivatives(F, X, weights, coeff):
 
    g0 = astroConstants(4) # [km/s**2]
    gravitational_parameter = astroConstants(13)
    [Isp, eta, P0] = satellite_data()[2:5]
    D = satellite_data()[8] # no degradation effect
    
    a = X[0]
    h = X[1]
    k = X[2]
    p = X[3]
    q = X[4]
    mass = X[5]

    # Initial values
    equinoctial_elements_initial = satellite_data()[0]
    keplerian_elements_initial = eq2kep(equinoctial_elements_initial)

    # Current values
    equinoctial_elements = EquinoctialElements(a, h, k, p, q, F)
    keplerian_elements = eq2kep(equinoctial_elements)
    if (keplerian_elements.e >= 1): # check if the orbit becomes a parabola = no longer an ellipse
       return np.inf
    E = F - keplerian_elements.raan - keplerian_elements.omega
    theta = 2*np.arctan(np.sqrt((1 + keplerian_elements.e)/(1 - keplerian_elements.e)) * np.tan(E/2))

    # Final values
    equinoctial_elements_final = satellite_data()[6]
    keplerian_elements_final = eq2kep(equinoctial_elements_final)
    delta_a_current = keplerian_elements_final.a - keplerian_elements.a # [adim] final - current semi-major axis
    delta_e_current = keplerian_elements_final.e - keplerian_elements.e # [adim] final - current eccentricity
    delta_i_current = keplerian_elements_final.i - keplerian_elements.i # [rad] final - current inclination
    delta_a_total = keplerian_elements_final.a - keplerian_elements_initial.a # [adim] final - initial semi-major axis
    delta_e_total = keplerian_elements_final.e - keplerian_elements_initial.e # [adim] final - initial eccentricity
    delta_i_total = keplerian_elements_final.i - keplerian_elements_initial.i # [rad] final - initial inclination

    # Weights (Kitamura approach)
    weight_a = weights[0] * delta_a_current/abs(delta_a_total)
    weight_e = weights[1] * delta_e_current/abs(delta_e_total)
    weight_i = weights[2] * delta_i_current/abs(delta_i_total)
    
    # Evaluation of the first derivatives of the equinoctial orbital elements
    sF = np.sin(F)
    cF = np.cos(F)

    G = np.sqrt(1 - k**2 - h**2)
    b = 1/(1 + G)
    n = np.sqrt(gravitational_parameter/(a**3))
    r = a * (1 - k*cF - h*sF)
    K = 1 + p**2 + q**2

    X = a * ((1 - (h**2)*b)*cF + h*k*b*sF - k)
    Y = a * ((1 - (k**2)*b)*sF + h*k*b*cF - h)
    dX = (a**2)*n/r * (h*k*b*cF - (1 - (h**2)*b)*sF)
    dY = (a**2)*n/r * ((1 - (k**2)*b)*cF - h*k*b*sF)

    dX_dh = a * (-(h*cF - k*sF)*(b + (h**2)*(b**3)/(1 - b)) - a/r*cF*(h*b - sF))
    dX_dk = -a * ( (h*cF - k*sF)*h*k*(b**3)/(1 - b) + 1 + a/r*sF*(sF - h*b) )

    dY_dh = a * ( (h*cF - k*sF)*h*k*(b**3)/(1 - b) - 1 + a/r*cF*(k*b - cF) )
    dY_dk = a * ( (h*cF - k*sF)*(b + (k**2)*(b**3)/(1 - b)) + a/r*sF*(cF - k*b) )

    d_a = [2*a/(n*r) * (h*k*b*cF - (1 - b*(h**2))*sF), 2*a/(n*r) * ((1 - (k**2)*b)*cF - h*k*b*sF), 0]
    d_h = [G/(n*(a**2)) * (dX_dk - h*b*dX/n), G/(n*(a**2)) * (dY_dk - h*b*dY/n), k/(G*n*(a**2)) * (q*Y - p*X)]
    d_k = [- G/(n*(a**2)) * (dX_dh + k*b*dX/n), - G/(n*(a**2)) * (dY_dh + k*b*dY/n), - h/(G*n*(a**2)) * (q*Y - p*X)]
    d_p = [0, 0, K*Y/(2*G*n*(a**2))]
    d_q = [0, 0, K*X/(2*G*n*(a**2))]
    M = [d_a, d_h, d_k, d_p, d_q]

    # Angles
    theta_a = 0
    s_theta_e = r*np.sin(theta) / np.sqrt( (r*np.sin(theta))**2 + (2*a*(keplerian_elements.e + np.cos(theta)))**2 )
    c_theta_e = 2*a*(keplerian_elements.e + np.cos(theta)) / np.sqrt( (r*np.sin(theta))**2 + (2*a*(keplerian_elements.e + np.cos(theta)))**2 )
    theta_e = np.arctan2(s_theta_e, c_theta_e)

    s_gamma = keplerian_elements.e*np.sin(theta) / np.sqrt( (keplerian_elements.e*np.sin(theta))**2 + ((1 + keplerian_elements.e*np.cos(theta)))**2 )
    c_gamma = (1 + keplerian_elements.e*np.cos(theta)) / np.sqrt( (keplerian_elements.e*np.sin(theta))**2 + ((1 + keplerian_elements.e*np.cos(theta)))**2 )
    gamma = np.arctan2(s_gamma, c_gamma)

    # Direction of the thrust
    u_a = np.array([np.sin(theta_a + gamma), np.cos(theta_a + gamma), 0])
    u_e = np.array([np.sin(theta_e + gamma), np.cos(theta_e + gamma), 0])
    u_i = np.array([0, 0, np.pi/2 * (np.cos(keplerian_elements.omega + theta))])

    u = np.dot(weight_a, u_a) + np.dot(weight_e, u_e) + np.dot(weight_i, u_i)
    u = u/np.linalg.norm([u[0], u[1], u[2]])

    delta = coeff * np.arctan2(u[0], u[1])
    sigma = coeff * np.arcsin(u[2])
    
    dr = np.sqrt(dX**2 + dY**2)

    a1 = np.cos(delta - gamma)*np.cos(sigma)*dX/dr + np.sin(delta - gamma)*np.cos(sigma)*dY/dr
    a2 = - np.sin(delta - gamma)*np.cos(sigma)*dX/dr + np.cos(delta - gamma)*np.cos(sigma)*dY/dr
    a3 = np.sin(sigma)
    
    alpha_hat = np.array([a1, a2, a3])

    # First derivative
    dF = (1 - k*cF - h*sF)/(2*np.pi)
    a_T = 2*eta*P0*D / (mass*g0*Isp) # thrust magnitude
    d_eq = np.dot(a_T*dF, M @ alpha_hat) # first derivatives of the equinoctial vector
    d_m = [-2*eta*P0*D/((g0*Isp)**2) * dF] # first derivative of the mass

    first_derivatives = np.concatenate((d_eq, d_m), axis = 0)

    return first_derivatives