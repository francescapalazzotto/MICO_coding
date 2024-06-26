'''
    COMPITO 21 GIUGNO 2024
    Una stazione televisiva ha a disposizione 1 ora in totale per trasmettere 
    spot pubblicitari, di durate diverse e pagati in maniera diversa (di valore diverso). 
    Deve quindi scegliere quali spot mandare in onda massimizzando il profitto 
    e l'occupazione del tempo a disposizione.
    Si risolva il problema usando un algoritmo genetico e si confronti 
    la soluzione con quella ottenuta con il simulated annealing.
    
    Parte Opzionale: Una volta scelti gli spot si faccia in modo da dividerli 
    in slot di 10 minuti cercando di minimizzare il numero di slot totali.
'''
from collections import namedtuple
import math
import random


# FUNZIONE PER L'ANALISI DEI DATI ANALIZZATI
def dataAnalysis( dataset: type[tuple[any, ...]] ) -> tuple:
    '''
    Funzione progettata per fare un'analisi dei dati a disposizione.
    Inserire in input il dataset riorganizzato all'interno di una namedtuple
    per analizzare e studiare i dati a disposizione.
    Inserire un campo nominato "cost" per l'analisi del costo di ogni 
    dato con tipo di dato un intero.
 
    Ritorna una tupla contente in ordine le seguenti informazioni:
        - massimo 
        - minimo
        - media aritmetica
        - totale
        - fitness: 1/somma dei costi
    '''
    
    # Raccoglie in una lista tutti i costi dei soggetti analizzati 
    costRates = [
        data.cost for data in dataset
    ]

    timeRates = [
        data.time for data in dataset
    ]
    
    # Calcolo delle metriche di interesse rispetto ai costi
    maxCost = max(costRates)
    minCost = min(costRates)
    totCost = sum(costRates)
    averageCost = totCost/len(costRates) if len(costRates) > 0 else 0
    fitnessCost = 1/totCost if totCost != 0 else float('inf')

    # Calcolo delle metriche di interesse rispetto ai tempi
    maxTime = max(timeRates)
    minTime = min(timeRates)
    totTime = sum(timeRates)
    averageTime = totTime/len(timeRates) if len(timeRates) > 0 else 0
    fitnessTime = 1/totTime if totTime != 0 else float('inf')


    return maxCost, minCost, averageCost, totCost, fitnessCost, maxTime, minTime, totTime, averageTime, fitnessTime

# FUNZIONE PER LA CREAZIONE DI UNA SOLUZIONE RANDOM
def createRandomSolution( individuals: list, k: int ) -> list:
    '''
    Funzione per creare una soluzione randomica binaria.
    Sfrutta lo score creato per valutare se la soluzione è ottimale
    sulla base della funzione obiettivo.
    Viene inserita la proprietà Info per fare riferimento alla caratteristica
    della funzione obiettivo da dover considerare.

    Input:
        - individuals: lista degli individuui rappresentanti il problema
        - k: intero che indica il numero ideale di individui scelti.
    '''
    while True:

        # Genero k indici casuali tra tutti i soggetti 
        positions = random.sample(range(len(individuals)), k)
        randomSolution = [
            True if i in positions else False for i in range(len(individuals)) 
        ]

        randomTime = sum(
            individuals[i].time for i, el in enumerate(randomSolution) if el
        )

        if randomTime <= 3600:
            return randomSolution

# GREEDY APPROACH
# Implementazione dell'algoritmo dinamico applicato al Knapsack problem
def find_max_value( objects : list , maximum_weight : int ) -> tuple[int, list]:
    '''
    Implementation of the algorithm using dynamic programming in order to determine,
    given the list of objects and the maximum weight carried by the sack,
    the maximum value/benefit of the objects that will be contained.

    Input:
        - list of objects: couples (weight, value)
        - maximum_weight: maximum weight of the knapsack 
    
    Output:
        - maximum value/benefit
        - table (needed to know which objects are in the sack)
    '''
    # Ordering the objects with increasing weight
    objects = sorted(objects, key = lambda x: x.time)


    # Implementation of the matrix of size (number of objects)x(maximum weight)
    n = len(objects)    # Number of objects
    table = []          # Using a list of lists to contain the values of the matrix

    # Filling the first row with 0s
    table.append( [0] * (maximum_weight + 1) )

    # Filling the rest of the table in which also the first column has 0s
    # Each row represent an object: going through the objects, it fills the rows
    for i in range(1, n + 1):

        obj = objects[i-1]    # Taking the (i-1)-th object for the i-th row
        row = [0]           # Implementation of the first column at the beginning
        for w in range(1, maximum_weight + 1):
           
            # Check if the weight of the object selected is heavier than the current weight (column)
            if obj[0] <= w:

                # Check if the value of the possible new object is greater then previous one:
                # if so, it adds the value in the table;
                # if not, it adds the previous one (upper cell)
                if obj[1] + table[i-1][w - obj[0]] > table[i-1][w]:
                    row.insert( w , obj[1] + table[i-1][w-obj[0]] )
                else:
                    row.insert( w , table[i-1][w] )
            else:
                row.insert( w , table[i-1][w] )
        
        table.append( row )
    
    # Take the number desidered: the maximum value in the 'sack
    K = table[n][maximum_weight] 
    
    return K, table

def find_objects( table : list, maximum_weight : int, objects : list, K : int ) -> list:
    '''
    This function uses the table contructed with the function "find_number_objects"
    in order to determine which objects we choose to put into the 'sack.
    
    Input:
        - table: the matrix/list of lists of the function containing the values
        - maximum_weight: maximum weight of the knapsack 
        - objects: sorted list of the objects in respect to the weight
        - K: maximum value to put into the 'sack
    
    Output:
        - list of objects to insert in the 'sack
    '''
    # Ordering the objects with increasing weight
    objects = sorted(objects, key = lambda x: x.time)

    # Consider two parameters to traverse the table: it starts from the right-bottom position
    i = len(table)-1        # Number of rows = total number of objects
    w = maximum_weight      # Last column of the table

    # Using a list to insert the objects taken from the list
    chosen_obj =[]

    while (i > 0) and (K > 0):

        # Check if the previous element of the column is the same
        if table[i][w] != table[i-1][w]:
            chosen_obj.append(objects[i-1])
            i -= 1
            K -= objects[i-1][0]
        else:
            i -= 1
    
    return chosen_obj

def greedySpots(
        setSpots: list[tuple[int, float]],
        maxTime: int
) -> list[any]:
    
    maxValue, matrix = find_max_value( setSpots, maxTime )
    return find_objects( matrix, maxTime, setSpots, maxValue )

# FUNZIONE DI ENERGY
# Si valuta la qualità della soluzione sulla base del costo totale ricavato
# dalla messa in onda degli spot selezionati nella soluzione. 
# Tale costo dovrà essere massimizzato.
def qualitySpots(
        xSolution: list[bool],
        setSpots: list[any]
) -> float:
    
    # Scorro la lista della soluzione attuale e sommo tutti i corrispettivi 
    # costi per valutare il costo totale presente
    totalCost = 0
    for index, el in enumerate(xSolution):
        if el:
            totalCost += setSpots[index].cost
    
    return totalCost

# TWEAK OPERATOR
# Funzione di modifica della soluzione: si attua una modifica randomica di 
# uno degli spot inserito all'interno della soluzione, eseguendo un controllo
# sul tempo massimo totale da rispettare.
def tweakSpots(
        xSolution: list[bool],
        setSpots: list[any],
        maxTime: int
) -> list[bool]:
    '''
    Funzione per modificare la soluzione corrente randomicamente e valutando
    la modifica in base al valore dello score. 

    Input: 
        - xSolution: soluzione attuale binaria da modificare.
        - individuals: lista di tutti gli individui per accedere al loro costo.
        - maxTime: vincolo da rispettare per la modifica della soluzione.

    Output:
        - lista binaria rappresentatnte la soluzione modificata.
    '''

    while True:

        xTweaked = xSolution.copy()
        
        # Choosing a random index to change in the current solution
        index = random.sample(range(len(xSolution)-1), 3)

        # Changing the corresponding position in the solution
        # If it is False, change it in True; and viceversa
        xTweaked[index[0]] = not xTweaked[index[0]]
        xTweaked[index[1]] = not xTweaked[index[1]]
        xTweaked[index[2]] = not xTweaked[index[2]]
    
        # Checking if the new solution generated is acceptable
        xTweakedTime = sum(
            setSpots[i].time for i, el in enumerate(xTweaked) if el
        )

        # Evaluate the score of this tweaked solution through the dimensions (time)
        if xTweakedTime <= maxTime:
            return xTweaked

# SIMULATED ANNEALING ALGORITHM
def simulatedAnnealingSpots(
        initialSolution: list[bool],
        initialTemperature: float,
        minTemperature: float,
        maxTime: int,
        alpha: float,
        population: list[any]
) -> list[bool]:
    '''
    Funzione che applica l'algoritmo di simulated annealing utilizzando
    una cooling schedule di tipo statico-dinamico.

    Input:
        - initialSolution: soluzione iniziale
        - initialTemperature: temperatura iniziale
        - minTemperature: temperatura minima finale
        - maxTime: tempo massimo della stazione televisiva
        - alpha: velocità di raffreddamento
        - population: lista di tutti gli individui del problema
    '''

    # Calcolo del numero di steps della cooling schedule utilizzando
    # il parametro tau in funzione del parametro alpha
    tau = (-1)/math.log(alpha)
    L = int(5*tau)

    # Inizializzazione dei parametri
    currentSolution = initialSolution
    temp = initialTemperature
    best = initialSolution

    # Decrescita lenta della temperatura fino alla temperatura minima stabilita
    while temp > minTemperature:

        # Thermal balance test
        for _ in range(L):

            # Modifica della soluzione corrente utilizzando il tweak operator
            copySolution = currentSolution.copy()
            candidateSolution = tweakSpots(copySolution, population, maxTime)

            # Calcolo del valore del delta: differenza della qualità tra 
            # la soluzione attuale e quella modificata
            delta = qualitySpots(candidateSolution, population) - qualitySpots(currentSolution, population)

            # Determinare se la soluzione candidata risulti accettabile 
            if delta < 0:
                # Soluzione accettabile
                currentSolution = candidateSolution
            else:
                num = random.uniform(0,1)
                pDelta = math.exp(-delta/temp)
                if num < pDelta:
                    # Soluzione accettata
                    currentSolution = candidateSolution

            if qualitySpots(currentSolution, population) > qualitySpots(best, population):
                best = currentSolution
            
        temp *= alpha

    return best

# FITNESS FUNCTION
# Funzione di fitness per la valutazione delle varie soluzioni che compongono 
# la popolazione iniziale di soluzioni.
def fitnessSpots(
        population: list[list[bool]],
        setSpots: list
) -> list[tuple[list[bool], float]]:
    
    fittedPopulation = [(ind, qualitySpots(ind, setSpots)) for ind in population]
    
    return sorted(fittedPopulation, key=lambda x: x[1], reverse=True)

# ISIDEAL FUNCTION
# Funzione di isIdeal per determinare se abbiamo raggiunto la soluzione ideale
def isIdealSpots(
        solution: list[bool],
        setSpots: list,
        targetFitness: float
) -> bool:
    
    return qualitySpots(solution, setSpots) <= targetFitness

# Funzione di ONE-POINT-CROSSOVER
def crossoverSpots(
        parentA: list[bool], 
        parentB: list[bool]
) -> tuple[list[bool], list[bool]]:
    
    # Implementa un crossover a punto singolo per i bin
    crossoverPoint = random.randint(0, len(parentA) - 1)
    childA = parentA[:crossoverPoint] + parentB[crossoverPoint:]
    childB = parentB[:crossoverPoint] + parentA[crossoverPoint:]
    
    return childA, childB

# Funzione di MUTAZIONE
def mutationSpots(
        individual: list[bool],
        setSpots: list[any], 
        mutationRate: float,
        maxTime: int
) -> list[bool]:
    
    if random.random() < mutationRate:
        return tweakSpots(individual, setSpots, maxTime)
    
    return individual

# Funzione di SELEZIONE degli individui della popolazione
def selectionSpots(
        population: list[tuple[list[bool], float]], 
        tournamentSize: int
) -> list[bool]:
    
    tournament = random.sample(population, tournamentSize)
    tournament = sorted(tournament, key=lambda x: x[1], reverse=True)
    
    return tournament[0][0]

# GENETIC ALGORITHM
# Implementazione dell'algoritmo genetico
def geneticAlgorithmSpots(
        sizePopulation: int,
        numGenerations: int,
        population: list[list[bool]],
        setSpots: list[any],
        tournamentSize: int = 2,
        targetFitness: int = 3600,
        mutationRate: float = 0.01,
        fitness: callable = fitnessSpots,
        isIdeal: callable = isIdealSpots,
        crossover: callable = crossoverSpots,
        mutation: callable = mutationSpots,
        selection: callable = selectionSpots
) -> list[bool]:
    '''
    Funzione che implementa l'algoritmo genetico.

    Input:
        - sizePopulation: dimensione della popolazione desiderata
        - numGenerations: numero di generazioni desiderate
        - population: popolazione di partenza
        - tournamentSize: dimensione del torneo per la selezione
        - targetFitness: fitness target desiderato
        - mutationRate: tasso di mutazione
        - fitness: funzione di valutazione di ogni individuo
        - isIdeal: funzione di valutazione della popolazione totale
        - crossover: funzione per l'applicazione del crossover
        - mutation: funzione per la mutazione degli individui
        - selection: funzione per la selezione degli individui
    
    Output:
        - lista degli individui ottenuti
    '''
    
    # Inizializzazione della variabile per la migliore soluzione trovata
    best = None

    # Implementazione del ciclo per le generazioni desiderate + valutazione
    # della best soluzione trovata.
    # Funzione isIdeal per la valutazione della fitness ottimale
    # In questo caso, implementiamo una funzione in cui la qualità totale dei bin
    # creati ha una qualità in termini di percentuali di riempimento superiore a 0.9
    for generation in range(numGenerations):

        fittedPopulation = fitness(population, setSpots)

        if best is None:
            best = fittedPopulation[0][0]
        
        for ind in fittedPopulation:
            if ind[1] > qualitySpots(best, setSpots):
                best = ind[0]
        
        if isIdeal(best, setSpots, targetFitness):
            break
        
        nextPopulation = []

        for _ in range(sizePopulation // 2):
            parentA = selection(fittedPopulation, tournamentSize)
            parentB = selection(fittedPopulation, tournamentSize)

            childA, childB = crossover(parentA, parentB)
            nextPopulation.append(mutation(childA, setSpots, mutationRate, targetFitness))
            nextPopulation.append(mutation(childB, setSpots, mutationRate, targetFitness))
        
        population = nextPopulation
    
    return best

if __name__ == '__main__':

    #* LETTURA E RACCOLTA DATI
    path = r"C:\Users\palaz\OneDrive\Desktop\University\UNIPA 2.0\II ANNO\Semestre 1\MICO - Machine Intelligence for Combinatorial Optimisation\MICO_coding\Data\Compito_21_06\table.txt"
    with open(path, mode="r") as dataset:
        lines = dataset.readlines()
    
    # Pulizia dei dati estrapolati
    # Divisione dei contenuti tra nome spot, costo e durata
    lines = [
        l.strip().split('\t') for l in lines[1:]
    ]

    # Utilizzo di una namedtuple per immagazzinare tutte informazioni:
    # nome e costo(dimensione) di ogni file considerato
    Spot = namedtuple( 'Spot', ['nome', 'time', 'cost'])
    spots = []
    for l in lines[:-1]:
        if l[1] == '1':
            spots.append(
                Spot(int(l[0])-1, 
                     60+int(l[2]), 
                     float(l[3][4:]))
            )
        else:
            spots.append(
                Spot(int(l[0])-1, 
                     int(l[2]), 
                     float(l[3][4:]))
            )
    
    maxCost, minCost, meanCost, totCost, fitnessCost, maxTime, minTime, totTime, averageTime, fitnessTime = dataAnalysis(spots)

    print("---- SPOT TELEVISIVI ----")
    print("ANALISI DEI DATI")
    print(f"Costo massimo: {maxCost}")
    print(f"Costo minimo: {minCost}")
    print(f"Costo media: {meanCost}")
    print(f"Totale somma costi: {totCost}")
    print(f"Fitness dei costi: {fitnessCost}")
    print()
    print(f"Tempo massimo: {maxTime}")
    print(f"Tempo minimo: {minTime}")
    print(f"Tempo media: {averageTime}")
    print(f"Totale somma tempi: {totTime}")
    print(f"Fitness dei tempi: {fitnessTime}")
    print()
    print("Considerazioni:")
    print(f"Sulla base delle analisi, il numero ideale di spot da mandare in onda è pari a {math.ceil(3600/averageTime)}")
    print("----------------------")
    
    #* GREEDY APPROACH SOLUTION
    print("-- GREEDY ALGORITHM --")
    greedySolution = greedySpots(spots, 3600)
    print(f"Numero spot selezionati {len(greedySolution)}:")
    # for s in greedySolution:
    #     print(s)
    print(f"Costo totale: {sum(el.cost for el in greedySolution)}")
    print(f"Durata media spot: {sum(el.time for el in greedySolution)/len(greedySolution)}")
    print(f"Tempo totale: {sum(el.time for el in greedySolution)}")
    print("----------------------")

    # Trasformazione della soluzione greedy in soluzione binaria
    greedySolutionBin = [False]*len(spots)
    for el in greedySolution:
        greedySolutionBin[el.nome] = True

    # Creazione di una soluzione randomica
    randomSolution = createRandomSolution(spots, 55)

    #* SIMULATED ANNEALING SOLUTION
    deltaE = 55
    t0 = 20*deltaE
    tMin = deltaE/10

    print("- SIMULATED ANNEALING -")
    simulatedSolutionBin = simulatedAnnealingSpots(
        initialSolution=randomSolution,
        initialTemperature=t0,
        minTemperature=tMin,
        maxTime=3600,
        alpha=0.99,
        population=spots
    )
    simulatedSolution = []
    for ind, el in enumerate(simulatedSolutionBin):
        if el:
            simulatedSolution.append(spots[ind])
    print(f"Numero spot selezionati {len(simulatedSolution)}:")
    # for s in simulatedSolution:
    #     print(s)
    print(f"Costo totale: {sum(el.cost for el in simulatedSolution)}")
    print(f"Durata media spot: {sum(el.time for el in simulatedSolution)/len(simulatedSolution)}")
    print(f"Tempo totale: {sum(el.time for el in simulatedSolution)}")
    print("----------------------")

    #* GENETIC ALGORITHM SOLUTION
    sizePopulation = 50
    numGenerations = 5
    targetFitness = 3600  # Fitness target desiderato: tempo massimo
    mutationRate = 0.01
    tournamentSize = 5
    initialPopulation = [
        createRandomSolution(spots, 55) for _ in range(sizePopulation)
    ]

    print("-- GENETIC ALGORITHM --")
    geneticSolutionBin = geneticAlgorithmSpots(
        sizePopulation,
        numGenerations,
        initialPopulation,
        spots,
        tournamentSize,
        targetFitness,
        mutationRate
    )
    geneticSolution = []
    for ind, el in enumerate(geneticSolutionBin):
        if el:
            geneticSolution.append(spots[ind])
    print(f"Numero spot selezionati {len(geneticSolution)}:")
    # for s in geneticSolution:
    #     print(s)
    print(f"Costo totale: {sum(el.cost for el in geneticSolution)}")
    print(f"Durata media spot: {sum(el.time for el in geneticSolution)/len(geneticSolution)}")
    print(f"Tempo totale: {sum(el.time for el in geneticSolution)}")
    print("----------------------")
