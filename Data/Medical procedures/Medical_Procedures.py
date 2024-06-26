'''
    MEDICAL PROCEDURES ASSIGNMENT: 
        - P = 15 procedures
        - M = 30 doctors available with different rates
    
    DATA SET: each doctor is represented by a binary vector for each
    procedure: 1 - practice it, 0 - not practise it. Cost value: rate of the 
    doctor related to his procedures. 

'''
from collections import namedtuple
from math import log, exp 
import numpy as np 
import random

def score( individual : list ) -> int:
    '''
    Function used to assign a score to a certain individual
    based on the procedures it practise.

    Input:
        - individual: list containing the procedures

    Counts the number of zeros (procedures not
    covered) this kind of solution hold and return the sum.
    '''

    score = sum( 1 if el == 0 else 0 for el in individual )
    
    return score

# ---
# GREEDY TECHNIQUE
def greedy_doctors( set_of_individuals : list ) -> list:
    '''
    Function that uses greedy approach in order to maximize the number of 
    procedures practised and minimize the rate.

    Input:
        - set_of_individuals: list of the individuals of the problem
    
    Output:
        - list containing the solution set
    '''

    # Partition of the set of individuals
    solution_set = [] # Elements chosen as final solution
    tosee_set = set_of_individuals.copy() # Elements not already seen


    # Initialization of final solution - choosing the one with minimum cost
    tosee_set = sorted(tosee_set, key=lambda x: x.cost, reverse=True )
    solution_set.append( tosee_set.pop() )

    while True:

        # Extracting a candidate solution - choosing a doctor who practises
        # a procedure that is not already covered and with the
        # minimum cost, and removing it from the set of solutions to see
        curr_procedures = np.sum( [el.procedures for el in solution_set], axis=0 )
        for i in range(15):
            if curr_procedures[i] == 0:
                
                # Extracting all the doctors who cover the procedures 
                cover_doc = [ doc for doc in tosee_set if doc.procedures[i] != 0 ]
                
                # Extracting the doctor with minimum cost
                doc_cand = min( cover_doc, key=lambda x: x.cost )

                # Inserting in the final solution + removing from the set to see
                solution_set.append(doc_cand)
                tosee_set.remove(doc_cand)
                
                break
        
        # Breaking the while-loop when all the procedures are covered
        if 0 not in curr_procedures:
            break   


    return solution_set 

# ---
# SIMULATED ANNEALING

# Defining a tweak operator to be used in the SA algorithm
def tweak_op( x_current : list, individuals : list ) -> list:
    '''
    Tweaking operator that modifies the current solution randomly 
    so that the generated solution includes doctors covering all procedures. 
    If this does not happen, recalculate the change.

    Input:
        - x_current: boolean list containing the doctors currently chosen.
        - individuals: list containing all the individuals studied.

    Output:
        - list: x_current modified that always covers all the procedures.  
    '''

    while True:

        # Choosing a random index to change in the current solution
        index = random.randint(0,29)

        # Changing the corresponding doctor in the solution
        # If it is 0, change it in 1, and viceversa
        x_tweaked = x_current.copy()
        if x_tweaked[index] == 1:
            x_tweaked[index] = 0
        else:
            x_tweaked[index] = 1
        
        # Checking if the procedures are covered
        x_tweaked_procedures = []
        for i in range(30):

            # Check if i-th doctor is chosen
            # If so, find its procedures 
            if x_tweaked[i] == 1:
                x_tweaked_procedures.append( individuals[i].procedures )

        x_tweaked_procedures_tot = np.sum( [proc for proc in x_tweaked_procedures], axis=0)

        # Function score determines the number of zeros in the list
        # of procedures: no zeros needed in the list
        score_value = score(x_tweaked_procedures_tot)
        if score_value == 0:
            break
    
    return x_tweaked

# Defining a function to determine the energy (cost) of the solutions
# to be used in the SA algorithm in order to check if we obtained
# a new solution with a lower cost
def energy_cost( x_solution : list, individuals : list ) -> int:
    '''
    Function that determines the total cost of the solution examined.

    Input:
        - x_solution: binary solution list to check cost.
        - individuals: list containing all the individuals studied.
    
    Output:
        - total cost of the solution.
    '''
    
    # Obtaining the cost of each doctor chosen in the solution examined
    x_solution_cost = 0
    for i in range(30):

            # Check if i-th doctor is chosen
            # If so, find its procedures 
            if x_solution[i] == 1:

                x_solution_cost += individuals[i].cost
    
    return x_solution_cost
    
def simulated_annealing_doctors( initial_solution : tuple, 
                                 initial_temperature : float,
                                 min_temperature : float, 
                                 alpha : float,
                                 individuals : list ) -> list:
    '''
    Function that applies the simulatead annealing algorithm.

    Input:
        - initial_solution
        - initial_temperature: starting temperature calculated as the ratio
        between the average energy and 20K (k is the Boltzman costant)
        - alpha: float number less than 1, velocity of the cooling schedule
        - individuals: list containing all the individuals studied. 
    
    Output:
        - solution set: binary list containing the solution obtained
    '''

    # Setting parameters
    x_current = initial_solution
    temp = initial_temperature
    k = 10*(1.380649*(10**(-23)))
    
    # Calculating the number of steps for the thermal balance test
    L = int(-5 / log(alpha))

    while temp > min_temperature:
        
        # Thermal balance test - repeat L times
        for _ in range(L):

            # Generate a solution modifying the current one
            x_candidate = tweak_op(x_current, individuals)

            # Calculate the value of energies
            dE = energy_cost(x_candidate, individuals) - energy_cost(x_current, individuals)

            if dE < 0:
                # Accept the new solution
                x_current = x_candidate
            else:
                # Generate a random number in ]0,1[
                num = random.random()
                P_dE = exp( -dE / k*temp )

                if num < P_dE:
                    # Accept the solution
                    x_current = x_candidate
            
            # Update the temperature
            temp *= alpha
    
    return x_current

# EVOLUTIONARY ALGORITHM (1,3)
def evolutionary_algorithm(
        
    initial_population : list,
    mu : int,
    lam : int,
    t_max : int,
    individuals : list 

    ) -> list:
    '''
    Function that applies the evolutionary algorithm using lam (lambda) 
    random individuals and mu best individuals at each generation 
    to generate other new individuals.

    Input:
        - initial_population: a candidate initial list of solutions
        - mu: mu value
        - lam: lambda value
        - t-max: number of generations to use to found the solution
        - individuals: list containing all the individuals studied

    Output:
        - Best solution found in the entire population.  
    '''

    # Initialize a list containing all the population used for each generation
    population = initial_population

    # Initialize a list to store the best solution found
    best = []

    t = 0
    while t < t_max:

        # Check in the population the best individual by checking the fitness
        # of each individual: if it finds a better individual with a less 
        # totale cost, it becomes the new best one found
        for ind in population:

            if ( best == [] or energy_cost(ind, individuals) < energy_cost(best, individuals) ):
                best = ind
        
        # Ordering the current population based on their fitness value
        # Creates a new list containing 2-tuples of individuals and their
        # respective fitness value 
        Q = [ (ind, energy_cost(ind, individuals)) for ind in population ]
        Q.sort(key=lambda x: x[1], reverse=True)

        # Take the mu individuals with the best fitness values in the population
        Q = Q[:mu]

        # Eliminating all the individuals of the population and creating
        # new individuals mutating the best ones
        population = [ tweak_op(ind[0], individuals) for ind in Q for _ in range(int(lam/mu)) ]

        t += 1
    
    return best


if __name__ == '__main__':
    
    # READ DATA
    # Reading dataset of doctors' procedures and respective cost rates
    path = r"C:\Users\palaz\OneDrive\Desktop\University\UNIPA 2.0\II ANNO\Semestre 1\MICO - Machine Intelligence for Combinatorial Optimisation\MICO_coding\Data\Medical procedures\ING_data.txt"
    with open(path, mode="r") as dataset:
        lines = dataset.readlines()
    
    # Extracting the procedures for each doctor
    # Creating a list of length 30 (number of doctor)
    procedures = [] # It can be interpreted as a matrix 30x15
    for line in lines[0:30]:
        string = line.strip()
        string = string[1:44].split(sep=",")
        string = [ el.strip() for el in string ]
        proc = [ int(el) for el in string ]
        procedures.append(proc)
    
    # Extracting the cost values for each doctor
    string = lines[31]
    cost_rates = [ int(el.split(sep=",")[0]) for el in string[1:177].split() ]
    
    # Extracting information about the total number of procedures practised
    # by each doctor - using it as fitness/quality evaluation
    proc_tot = [ sum(procs) for procs in procedures ]


    #---
    # Creating the dataset containing all the information
    Doctor = namedtuple( 'Doctor', ['name', 'procedures', 'cost'])
    
    # List containing all the doctors name and their respective
    # procedures and cost
    doctors = [] 
    for i in range(30):
        doc = Doctor('doc' + str(i), np.array(procedures[i]), cost_rates[i])
        doctors.append(doc)
    

    # ---
    # START IMPLEMENTATION OF VISUALIZATIONS OF RESULTS
    print()
    print("MEDICAL PROCEDURE OPTIMIZATION")
    print("---")

    # Visualization of the possibile value of data owned
    print("DATA ANALYSIS")
    print(f"Minimum rate of doctors: {min(cost_rates)}")
    print(f"Maximum rate of doctors: {max(cost_rates)}")
    print(f"Average rate of doctors: {sum(cost_rates)/30}")
    print(f"Total rate of doctors: {sum(cost_rates)}")
    print(f"Average of procedures practised: {sum(proc_tot)/30}")
    print(f"Total fitness value: {1/sum(cost_rates)}")
    print(f"Average fitness value: {1/(sum(cost_rates)/30)}")
    print("---")

    #* Implementation of Greedy techique
    print("GREEDY ALGORITHM")
    greedy_solution = greedy_doctors(doctors)
    print("Doctors to choose are the following:")
    for doc in greedy_solution:
        print(f"{doc.name} | {doc.cost}")
    greedy_procedures = np.sum( [doc.procedures for doc in greedy_solution], axis=0)
    print(f"Procedures: {greedy_procedures}")
    print(f"Average rate: {sum(doc.cost for doc in greedy_solution)/len(greedy_solution)}")
    print(f"Total rate: {sum(doc.cost for doc in greedy_solution)}")
    print("---")

    # ---
    # Transforming the greedy solution in a boolean string in order to
    # study and apply the following algorithms
    # Constructed as: 1 - doctor chosen; 0 - otherwise
    x_greedy = [0]*30
    for doc in greedy_solution:
        index = int(doc.name[3:])
        x_greedy[index] = 1

    k = 10*(1.380649*(10**(-23)))
    # Calculate the average cost
    delta_E = sum(doc.cost for doc in greedy_solution)/len(greedy_solution)
   
    # Calculate the initial temperature
    T_0 = (20*delta_E) / k

    # Calculate the minimum temperature
    T_min = delta_E / k

    #* Implementing Simulated Annealing algorithm 
    print("SIMULATED ANNEALING ALGORITHM")
    sa_solution = simulated_annealing_doctors( initial_solution= x_greedy,
                                               initial_temperature= T_0,
                                               min_temperature= T_min,
                                               alpha= 0.99,
                                               individuals= doctors)
    sa_doctors = []
    for i in range(30):
        if sa_solution[i] == 1:
            for doc in doctors:
                if i == int(doc.name[3:]):
                    sa_doctors.append(doc)
                    print(f"{doc.name} | {doc.cost}")
                    break
    sa_procedures = np.sum( [doc.procedures for doc in sa_doctors ], axis=0)
    print(f"Procedures: {sa_procedures}")
    print(f"Average rate: {sum(doc.cost for doc in sa_doctors)/len(sa_doctors)}")
    print(f"Total rate: {sum(doc.cost for doc in sa_doctors)}")
    print("---")

    #* Implementing population method - (1,3)-Evolutionary Algorithm
    print("(1,3)-EVOLUTIONARY ALGORITHM")

    # Generate a random solutions to start from: generates 3 random solutions
    # generated starting from the greedy solution and mutating it
    init_sol = [x_greedy]
    for _ in range(2):
        init_sol.append(tweak_op(x_greedy, doctors))

    ea_solution = evolutionary_algorithm(init_sol, 1, 5, 100, doctors)
    ea_doctors = []
    for i in range(30):
        if ea_solution[i] == 1:
            doc = doctors[i]
            ea_doctors.append(doc)
            print(f"{doc.name} | {doc.cost}")   
    ea_procedures = np.sum( [doc.procedures for doc in ea_doctors ], axis=0)
    print(f"Procedures: {ea_procedures}")
    print(f"Average rate: {sum(doc.cost for doc in ea_doctors)/len(ea_doctors)}")
    print(f"Total rate: {sum(doc.cost for doc in ea_doctors)}")
    print("---")