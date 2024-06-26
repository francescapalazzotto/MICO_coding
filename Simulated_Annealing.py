'''
    SIMULATED ANNEALING ALGORITHM
'''

import random, math

# Defining the energy function of the problem that will be the evaluation
# of the cost function under consideration
def energy_function( solution : any ) -> any:
    '''
    Function that determines the value of the objective function under 
    consideration. It could be of any type based on the representation.
    of the problem.

    Input:
        - solution: type based on the representation
    
    Output:
        - value: evaluation of the candidate solution
    '''

    return None

# Defining a function to modify a solution during the process
def modify( solution : any ) -> any:
    '''
    Function that modify a solution of the problem.
    This modification is chosen based on the configuration model.

    Input:
        - solution to be modified
    
    Output:
        - modification of the solution
    '''

    mod = function(solution) # Some kind of function applied to the solution

    return mod

# Types of the parameters depends on the representation of the problem
def simulated_annealing( initial_solution : any, 
                         initial_energy : any,
                         initial_temperature : any,
                         alpha : float ) -> list:
    
    '''
    Function that applies the simulated annealing function
    using an hybrid cooling schedule (static-dynamic).

    Input:
        - initial solution
        - initial energy: value of the cost function of the initial solution
        - initial temperature: a high number (tends to 0)
        - alpha: float number less than 1, velocity of the cooling schedule
    
    Output:
        - solutions set: list containing all the solution chosen
    '''

    # Calculating the number of steps of the cooling schedule
    # using the parameter tau in function of alpha
    tau = (-1) / math.log(alpha)
    L = 5 * tau
    
    # Initialization of the parameters
    sol_current = initial_solution
    temp = initial_temperature

    solutions_set = [] # It could be a single solution
    # Temperature decreases slowly until 0
    while temp > 0:

        # Thermal balance test
        for _ in range(L):
            
            # Tweaking the current solution
            sol_copy = sol_current.copy()
            sol_candidate = modify(sol_copy)

            # Calculate the value of delta: difference between the energies
            # of the current solution and the modified one
            delta = energy_function(sol_candidate) - energy_function(sol_current)

            # Determine if the candidate solution is acceptable
            if delta < 0:

                # Accepted
                solutions_set.append(sol_candidate)
                sol_current = sol_candidate

            else:

                num = random.uniform(0,1) # INSERIRE CONDIZIONE CHE MI AIUTI A NON PRENDERE 0
                p_delta = math.exp( -delta/( (1.380649*(10**(-23))) * temp ) )
                if num < p_delta:

                    # Accept the solution
                    solutions_set.append(sol_candidate)
                    sol_current = sol_candidate

        # Cooling schedule: static
        temp *= alpha 

    return solutions_set     