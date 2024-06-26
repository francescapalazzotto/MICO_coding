'''
    ALGORITHMS LIBRARY
    This file contains all the functions of the algorithms studied in course of Machine Intelligence for Combinatorial Optimization course.
'''

# Imported libraries
import random

# SAMPLE OF THE OBJECTIVE VALUE FUNCTION 
def objective_value( objective : any, individual : any) -> any:
    '''
    Function used to determine if an individual solution is good in an objective defined from the problem studied. 
    Input:
        - objective: constraint of the problem, could be of any type based on the representation of the problem.
        - individual: could be of any type based on the representation of the problem.
    Output:
        - any kind based on the criteria used to evaluate the goodness of the solution.  
    '''

    return None  

# In this function, it is used an objective value function that is different for each problem.
# It must be defined before using the following function.
def pareto_domination( A : any, B : any, objectives : list ) -> bool:
    '''
    Function used to determine if the individual solution A dominates the individual solution B,
    in relation to the objectives to assess of the problem considered. 
    Input:
        - A: solution of any type, based on the representation of the problem.
        - B: solution of any type, based on the representation of the problem.  
        - objectives: list containing all the constraints that the solution respects.
    Output:
        - bool: True - if A dominates B; False - if A does not dominate B.
    '''
    a = False
    for obj in objectives:
        if objective_value(obj, A) > objective_value(obj, B):
            a = True                                            # A might dominate B
        elif objective_value(obj, B) > objective_value(obj, A):
            return False                                        # A definitely does not dominate B
    return a

def pareto_front( individuals : list, objectives : list ) -> list:
    '''
    Function used to determine the Pareto Front of a population: a set of all pareto optimal solutions.
    Input:
        - individuals: list of individuals to compute the front among (the population)
        - objectives: list containing all the constraints that the solution respects.
    Output:
        - front: list containing all the pareto optimal solutions
    '''
    front = []
    for ind in individuals:
        front.append(ind)
        for el in front:
            if el != ind:
                # If there is an element which dominates the new element added to the front,
                # then the new elements is NOT pareto optimal.
                if pareto_domination(el, ind, objectives):
                    front.remove(ind)
                    break

                # If the new element dominates a front individual,
                # then the latter should be removed. 
                elif pareto_domination(ind, el, objectives):
                    front.remove(el)
    return front

def pareto_front_ranks( population : list, objectives : list ) -> list:
    '''
    Function used to determine and store all the pareto front ranks of the population studied.
    Input:
        - population: list of individuals studied.
        - objectives: list containing all the constraints that the solution respects.
    Output:
        - list of lists containing at each position i the list of individuals of pareto front rank i+1.
    '''
    individuals = population                        # Gradually remove individuals from it, without modifying the original population
    ranks = []
    i = 0 
    while individuals != []:
        
        # Assign at position i the individuals
        ranks[i] = pareto_front(individuals, objectives)

        # Remove the current front from the individuals 
        for el in ranks[i]:
            individuals.remove(el)
        
        i += 1
    return ranks

def fixed_length_vector( min : int, max : int, l : int ) -> list:
    '''
    Function used to generate a random real-valued vector of length l fixed and range of values desired.
    Input:
        - min: minimum of the element value.
        - max: maximum of the element value.
        - l: lenght of the list.
    Output:
        - list: containing real-value between min and max of length l.
    '''    
    vector = []
    for i in range(l):
        vector.append( round( random.uniform(min, max), 2 ) )
    return vector

'''
    LOCAL SEARCH FAMILITY ALGORITHMS
'''

# SAMPLE FOR THE GENERATION OF AN INITIAL SOLUTION - to adapt based on the mechanism we want to follow; 
#                                 TWEAKING OPERATOR - to adapt based on which kind of change/noise we want to introduce.
#                                 QUALITY - to adapt based on which solution is better to another
def initial_solution():
    return None

def tweaking_operator( solution : any ) -> any:
    solution = "..."
    return solution

def quality_solution( solution : any ) -> any:
    # it could be for example the distances
    distance = 2*solution
    return distance

# To insert which is the condition to be an ideal solution
def hill_climbing( initial_solution : any, steps = 20 ) -> any:
    '''
    Function used to determine the ideal solution in a number of steps, default 20, or less, introducing noise to the initial solution given.
    Input:
        - initial solution: starting from an approximate solution on your choice.
    Output:
        - ideal solution found.  
    '''
    time = 0
    return None
    