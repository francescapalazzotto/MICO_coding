'''
    ON-LINE JOURNAL ASSIGNMENT
    An online newspaper has 10Mbytes of space available for its articles and must
    maximize the number of potential readers. Each published article has an author
    with a different number of followers, the newspaper must therefore publish,
    respecting the space available, articles written by authors with the maximum
    number of followers, trying to maximize the number of readers.

'''
from collections import namedtuple
from math import log, exp  
import random

# ---
# GREEDY ALGORITHM
def journal_greedy( authors : list, total_space : int) -> list:
    '''
    Function used to implement a simple greedy algorithm to find a solution 
    of the problem and getting an idea of what we are looking for. 
    It simply gets all the author with maximum followers until the available 
    space is finished (or almost).

    Input: 
        - authors: list containing all the authors available to publish.
        - total_space: bytes available on the on-line newspaper.
    
    Output:
        - list containing all the authors chosen.
    '''

    # Ordering the list of authors in an decreasing order 
    authors = sorted(authors, key=lambda x: x.followers, reverse=True)

    # Initializing a list containing the authors chosen
    sol_greedy = []

    space_available = total_space
    for athr in authors:
        
        # Checking if the space occupied from the current author's article
        # runs out the available space or the latter is not enough.
        # If so, consider the current author's article; if not, stop the loop.
        if space_available - athr.dimension >= 0:
            sol_greedy.append(athr)
            space_available -= athr.dimension
        else:

            # Break the loop: space available not enough or has run out. 
            break
    
    return sol_greedy

# ---
# SIMULATED ANNEALING

# Definining a score function to use to evaluate the goodness of the solution
# It is related to the total dimension/space available of the on-line newspaper
def journal_score( total_dimensions : list ) -> int:
    '''
    Function to calculate the score of a solution calculating the total 
    dimensions occupied by the individuals chosen and determing the remaining
    from the total space available.

    Input:
        - total_dimensions: list containing all the articles' dimension
        of the current solution to be evaluated
    
    Output:
        - value of the remaining space available. It should not be negative.
    '''

    return 10000000 - sum(total_dimensions) 

# Defining a tweak operator to modify a solution 
def journal_tweak( x_solution : list, individuals : list ) -> list:
    '''
    Function to modify a current solution randomly and evaluating the 
    resulting modification through the score to obtain an acceptable
    solution.

    Input:
        - x_solution: current binary candidate solution to be modified.
        - individuals: list of the total individuals to access to their costs.

    Output:
        - binary list representing the solution tweaked.
    '''

    
    while True:

        x_tweaked = x_solution.copy()
        
        # Choosing a random index to change in the current solution
        index = random.randint(0,len(x_solution)-1)

        # Changing the corresponding position in the solution
        # If it is 0, change it in 1; and viceversa
        if x_tweaked[index] == 1:
            x_tweaked[index] = 0
        else:
            x_tweaked[index] = 1
    
        # Checking if the new solution generated is acceptable in respect to
        # the available space - determing all the articles' dimension of the solution
        x_tweaked_dimensions = []
        for i in range(len(x_solution)):

            # Consider the i-th author chosen in the tweaked solution
            if x_tweaked[i] == 1:

                x_tweaked_dimensions.append(individuals[i-1].dimension)
        
        # Evaluate the score of this tweaked solution through the dimensions
        tweaked_solution_score = journal_score(x_tweaked_dimensions)
        if tweaked_solution_score >= 0:
            break
    
    return x_tweaked

# Defining a function to evaluate the energy value
def journal_energy( x_solution : list, individuals : list ) -> int:
    '''
    Function to determine the total number of followers of a solution
    based on the authors chosen.

    Input:
        - x_solution: binary list containing the authors chosen.
        - individuals: list containing all the authors available.

    Output:
        - total number of the followers.
    '''

    # Determing all the followers of the solution considered
    x_solution_cost = 0
    for i in range(len(x_solution)):

        # Consider the i-th author chosen in the tweaked solution
        if x_solution[i] == 1:

            x_solution_cost += individuals[i-1].followers
    
    return x_solution_cost


# Implementation of the simulated annealing algorithm
def journal_simulated_annealing( initial_solution : list,
                                 initial_temperature : float,
                                 min_temperature : float,
                                 alpha : float,
                                 individuals : list) -> list:
    '''
    Function that implements the Simulated Annealing algorithm.

    Input:
        - initial_solution: binary list containing the authors chosen. 
        - initial_temperature: starting temperature calculated as the ratio
        between the average energy and 20K (k is the Boltzman costant)
        - alpha: float number less than 1, velocity of the cooling schedule
        - individuals: list containing all the individuals studied. 
    
    Output:
        - solution set: binary list containing the solution obtained.

    ''' 

    # Setting parameters
    x_current = initial_solution
    temp = initial_temperature
    k = 10*(1.380649*(10**(-23)))
    # Calculating the number of steps of the thermal balance test
    L = int(-5 / log(alpha))

    # Initializing a variable to store the best solution found
    best = initial_solution.copy()

    while temp > min_temperature:

        # Thermal balance test - repeat L times
        for _ in range(L):

            x_candidate = journal_tweak(x_current, individuals)

            # Calculate the value of energies - delta E
            # dE = journal_energy(x_candidate, individuals) - journal_energy(x_current, individuals)
            dE = journal_energy(x_current, individuals) - journal_energy(x_candidate, individuals)

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
            temp *= alpha*k

            # Checking if the the current solution found is better than the 
            # best found: if yes, replace it
            if journal_energy(x_current, individuals) > journal_energy(best, individuals):
                best = x_current


    return best

# ---
# POPULATION METHOD - ()-Evolution Strategy
def journal_evolutionary_algorithm(

    initial_population: list,
    mu: int,
    lam: int,
    t_max: int,
    individuals: list    

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

            if ( best == [] or journal_energy(ind, individuals) > journal_energy(best, individuals) ):
                best = ind
        
        # Ordering the current population based on their fitness value
        # Creates a new list containing 2-tuples of individuals and their
        # respective fitness value 
        Q = [ (ind, journal_energy(ind, individuals)) for ind in population ]
        Q.sort(key=lambda x: x[1], reverse=True)

        # Take the mu individuals with the best fitness values in the population
        Q = Q[:mu]

        # Eliminating all the individuals of the population and creating
        # new individuals mutating the best ones
        population = [ journal_tweak(ind[0], individuals) for ind in Q for _ in range(int(lam/mu)) ]

        t += 1
    
    return best


if __name__ == "__main__":

    print()
    print("ON-LINE JOURNAL OPTIMIZATION")
    print("---")

    #* DATA GAINING AND ANALYSIS
    # Reading the file containing all the information about the author 
    # of the on-line newspaper
    path = r"C:\Users\palaz\OneDrive\Desktop\University\UNIPA 2.0\II ANNO\Semestre 1\MICO - Machine Intelligence for Combinatorial Optimisation\MICO_coding\Data\Journal\ON-line_journal_DATAe.csv"
    with open( path, mode="r" ) as file:
        lines = file.readlines()
    lines = [ line.strip().split(sep=',') for line in lines[1:] ]
    
    # Implementing a namedtuple to store information
    Author = namedtuple('Author', ['name', 'dimension', 'followers'])
    authors = [ Author('A'+str(lines[i][0]), int(lines[i][1]), int(lines[i][2])) 
                                            for i in range(len(lines))  ]

    # Storing single information in different lists
    art_dim = [ ath.dimension for ath in authors ]
    author_foll = [ ath.followers for ath in authors ]


    # --- 
    # Data analysis:
    print("DATA ANALYSIS")
    print(f"Minimum article dimension: {min(art_dim)}")
    print(f"Maximum article dimension: {max(art_dim)}")
    print(f"Average article dimension: {sum(art_dim)/len(art_dim)}")
    print(f"Minimum author's followers: {min(author_foll)}")
    print(f"Maximum author's followers: {max(author_foll)}")
    print(f"Average author's followers: {sum(author_foll)/len(author_foll)}")
    # print(f"Average article's : {10000000/(sum(art_dim)/len(art_dim))}")
    print("---")
    
    #* GREEDY ALGORITHM
    print("GREEDY ALGORITHM")
    sol_greedy = journal_greedy(authors, 10000000)
    # print("Authors to choose are the following:")
    # for ath in sol_greedy:
    #     print(f"{ath.name}\t{ath.followers}")
    print(f"Total space occupied: {sum(ath.dimension for ath in sol_greedy)}")
    print(f"Total space unoccupied: {10000000-sum(ath.dimension for ath in sol_greedy)}")
    print(f"Total followers: {sum(ath.followers for ath in sol_greedy )}")
    print(f"Total authors: {len(sol_greedy)}")
    print("---")
    
    # Transformin the greedy solution in a boolean string in order to study
    # and apply the followings algorithms.
    # Constructed as 1 - author chosen; 0 - otherwise.
    binary_greedy = [0] * len(authors)
    for ath in sol_greedy:
        binary_greedy[ int(ath.name[1:])-1 ] = 1
    

    # Defining parameters to use in simulated annealing algorithm
    k = 10*(1.380649*(10**(-23)))
    # Average follower's cost
    delta_E = sum(ath.followers for ath in sol_greedy)/len(sol_greedy)
    # Calculate initial temperature 
    T_0 = (20*delta_E)/k
    # Calculate minimum temperature
    T_min = delta_E/(10*k)

    #* Implementing Simulated Annealing algorithm 
    print("SIMULATED ANNEALING ALGORITHM")
    print()
    print("Case 1 - starting from the Greedy solution:")
    SA_solution = journal_simulated_annealing( initial_solution=binary_greedy,
                                               initial_temperature=T_0,
                                               min_temperature=T_min,
                                               alpha=0.95,
                                               individuals=authors)
    SA_authors = []
    for i in range(len(SA_solution)):
        if SA_solution[i] == 1:
            # print(f"{authors[i-1].name}\t{authors[i-1].followers}")
            SA_authors.append(authors[i-1])
    print(f"Total space occupied: {sum(ath.dimension for ath in SA_authors)}")
    print(f"Total space unoccupied: {10000000-sum(ath.dimension for ath in SA_authors)}")
    print(f"Total followers: {sum(ath.followers for ath in SA_authors )}")
    print(f"Total authors: {len(SA_authors)}")
    print()

    print("Case 2 - starting from a random solution:")

    # Create a random solution checking the score of dimensions
    while True:

        random_solution = [0] * len(authors)
        positions = random.sample(range(200), 38)

        for pos in positions:
            random_solution[pos] = 1

        random_dimensions = []
        for i in range(len(random_solution)):
            if random_solution[i] == 1:
                random_dimensions.append( authors[i-1].dimension)
        
        if journal_score(random_dimensions) >= 0:
            break
    
    random_authors = []
    for i in range(len(random_solution)):

        # Consider the i-th author chosen in the random solution
        if random_solution[i] == 1:

            random_authors.append( authors[i-1] )

    # Average follower's cost
    delta_E_r = sum(ath.followers for ath in random_authors)/len(random_authors)
    # Calculate initial temperature 
    T_init = (20*delta_E_r)/k
    # Calculate minimum temperature
    T_min_r = delta_E_r/(10*k)

    SA_R_solution = journal_simulated_annealing( initial_solution=random_solution,
                                                 initial_temperature=T_init,
                                                 min_temperature=T_min_r,
                                                 alpha=0.95,
                                                 individuals=authors)
    SA_R_authors = []
    for i in range(len(SA_R_solution)):
        if SA_R_solution[i] == 1:
            # print(f"{authors[i-1].name}\t{authors[i-1].followers}")
            SA_R_authors.append(authors[i-1])
    print(f"Total space occupied: {sum(ath.dimension for ath in SA_R_authors)}")
    print(f"Total space unoccupied: {10000000-sum(ath.dimension for ath in SA_R_authors)}")
    print(f"Total followers: {sum(ath.followers for ath in SA_R_authors )}")
    print(f"Total authors: {len(SA_R_authors)}")
    print("---")

    #* Implementing Population algorithm 
    print("(5,20)-EVOLUTIONARY ALGORITHM")
    print()
    print("Case 1 - starting from the Greedy solution:")

    # Generate a random solutions to start from: generates 35 random solutions
    # generated starting from the greedy solution and mutating it
    init_sol = [binary_greedy]
    for _ in range(35):
        tweak_sol = journal_tweak(binary_greedy, authors)
        if tweak_sol not in init_sol:
            init_sol.append(tweak_sol)
    
    EA_solution = journal_evolutionary_algorithm( initial_population=init_sol,
                                                  mu=5,
                                                  lam=20,
                                                  t_max=20,
                                                  individuals=authors)
    EA_authors = []
    for i in range(len(EA_solution)):
        if EA_solution[i] == 1:
            # print(f"{authors[i-1].name}\t{authors[i-1].followers}")
            EA_authors.append(authors[i-1])
    print(f"Total space occupied: {sum(ath.dimension for ath in EA_authors)}")
    print(f"Total space unoccupied: {10000000-sum(ath.dimension for ath in EA_authors)}")
    print(f"Total followers: {sum(ath.followers for ath in EA_authors )}")
    print(f"Total authors: {len(EA_authors)}")
    print()

    print("Case 2 - starting from a random solution:")

    # Create a random solution checking the score of dimensions
    ran_solution = []
    for _ in range(35):
        while True:

            random_solution = [0] * len(authors)
            positions = random.sample(range(200), 38)

            for pos in positions:
                random_solution[pos] = 1

            random_dimensions = []
            for i in range(len(random_solution)):
                if random_solution[i] == 1:
                    random_dimensions.append( authors[i-1].dimension)
            
            if journal_score(random_dimensions) >= 0:
                ran_solution.append(random_solution)
                break
    
    EA_r_solution = journal_evolutionary_algorithm( initial_population=ran_solution,
                                                  mu=5,
                                                  lam=35,
                                                  t_max=20,
                                                  individuals=authors)
    EA_r_authors = []
    for i in range(len(EA_r_solution)):
        if EA_r_solution[i] == 1:
            # print(f"{authors[i-1].name}\t{authors[i-1].followers}")
            EA_r_authors.append(authors[i-1])
    print(f"Total space occupied: {sum(ath.dimension for ath in EA_r_authors)}")
    print(f"Total space unoccupied: {10000000-sum(ath.dimension for ath in EA_r_authors)}")
    print(f"Total followers: {sum(ath.followers for ath in EA_r_authors )}")
    print(f"Total authors: {len(EA_r_authors)}")
    print()
    
