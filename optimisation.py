
import numpy as np
import matplotlib.pyplot as plot
 
def particleSwarm(fun, LB, UB, toll, n_iterations, n_particles):

    # Convert the Lower and Upper boundaries into arrays
    LB = np.array(LB)
    UB = np.array(UB)
    n_variables = LB.size

    if (n_variables == 0):
        return print('ERROR : the problem must have at least on variable to be optimised. The optimisation function has stopped.')
    if (LB.size != UB.size):
        return print('ERROR : the size of the Boundaries do not match. The optimisation function has stopped.')
    for j in range(n_variables):
        if (UB[j] < LB[j]):
            return print('ERROR : the upper boundary has to be greater than (or at least equal to) the lower boundary. . The optimisation function has stopped.')

    # Hyper-parameter of the algorithm
    c1 = c2 = 0.1
    w = 0.8
    standard_deviation = 0.1

    # Create particles
    X = np.random.rand(n_variables, n_particles)
    for j in range(n_variables):
        X[j] = np.dot(X[j], (UB[j] - LB[j])) + LB[j] # takes the j-th vector (row) in matrix X
    V = np.random.randn(n_variables, n_particles) * standard_deviation

    # Initialize data
    pbest = X
    pbest_obj = []
    for j in range(n_particles):
        input_vector = X[0:n_variables, j] # initialisation of the vector of the inputs: row vector, number of rows = number of variables of the problem
        pbest_obj.append(fun(input_vector)) # 0, 1 sono le due variabili in ingresso

    pbest_obj = np.array(pbest_obj) # transform into an array
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    # Check if the tollerance is reached before using all the iterations available
    if (gbest_obj <= toll):
        return [gbest_obj, gbest]

    y_best = []
    for it in range(n_iterations):
        # Update parameters
        r1, r2 = np.random.rand(2) 
        V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1) - X)
        X = X + V

        obj = []
        for j in range(n_particles):
            for i in range(n_variables):
                if (X[i,j] < LB[i]):
                    X[i,j] = LB[i]
                if (X[i,j] > UB[i]):
                    X[i,j] = UB[i]

            input_vector = X[0:n_variables, j] # initialisation of the vector of the inputs: row vector, number of rows = number of variables of the problem
            obj.append(fun(input_vector))

        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj, obj]).min(axis = 0)
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()

        y_best.append(gbest_obj)
        # plot.figure(figsize = (4, 4))
        # plot.scatter(range(it+1), y_best, color = 'red')
        # plot.show()

        # Check if the tollerance is reached before using all the iterations available
        if (gbest_obj <= toll):
            return [gbest_obj, gbest]
        print('Iteration #', it, ': ', gbest_obj)
        
    return [gbest_obj, gbest]