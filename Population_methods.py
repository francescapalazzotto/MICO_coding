'''
    POPULATION METHODS (camelCase implementations)
'''
# Funzioni fittizie per il momento.
def energyCost( xSolution: list, individuals: list ) -> int: 
    return None

def tweakOp( xCurrent: list, individuals: list ) -> int: 
    return None

# Evolutionary Algorithm (mu, lambda)
# mu: numero di genitori scelti per creare le nuove generazioni - i migliori
# lambda: numero della popolazione iniziale 
def evolutionaryAlgorithmMuLam(
        
    initialPopulation: list,
    mu: int,
    lam: int,
    tMax: int,
    individuals: list   
     
    ) -> list:
    '''
    Function that applies the evolutionary algorithm using lam (lambda) 
    random individuals and mu best individuals at each generation 
    to generate other new individuals.

    Input:
        - initialPopulation: a candidate initial list of solutions
        - mu: mu value
        - lam: lambda value
        - tMax: number of generations to use to found the solution
        - individuals: list containing all the individuals studied

    Output:
        - Best solution found in the entire population. 
    '''

    # Initialize a list containing all the population used for each generation
    population = initialPopulation

    # Initialize a list to store the best solution found
    best = []

    t = 0
    while t <= tMax:

        # Check in the population the best individual by checking the fitness
        # of each individual of the candidate solution analysed: if it finds
        # a better individual with a less fitness rate, it becomes the new best 
        # one found
        for ind in population:
            if ( best == [] or energyCost(ind, individuals) < energyCost(best, individuals)):
                best = ind
        
        # Ordering the current population based on their fitness value
        # Creates a new list containing 2-tuples of individuals and their
        # respective fitness value 
        Q = [ (ind, energyCost(ind, individuals)) for ind in population ]
        Q.sort(key=lambda x: x[1], reverse=True)

        # Take the mu individuals with the best fitness values in the population
        Q = Q[:mu]

        # Eliminating all the individuals of the population and creating
        # new individuals mutating the best ones
        population = [ tweakOp(ind[0], individuals) for ind in Q for _ in range(int(lam/mu)) ]

        t += 1
    
    return best