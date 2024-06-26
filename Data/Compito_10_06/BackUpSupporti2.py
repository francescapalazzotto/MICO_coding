'''
    COMPITO 10 GIUGNO 2024
    Backup di una macchina contenente diversi file, i file sono 
    gia' stati disposti in un elenco con la loro dimensione 
    e sono riportati nel file dataset.txt.
    Il backup sara' eseguito su supporti ottici di capacita', 
    anch'essa indicata nel file dataset.txt.

    Si riporti come risultato:
    - numero m di supporti utilizzati
    - lo spazio libero rimasto su ciascun supporto
    - la lista dei file per ciascun supporto
'''

from collections import namedtuple
import numpy as np
import math
import random

# DATA ANALYSIS: sfruttare libreria namedtuple.
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
    
    # Calcolo delle metriche di interesse
    maxCost = max(costRates)
    minCost = min(costRates)
    totCost = sum(costRates)
    averageCost = totCost/len(costRates) if len(costRates) > 0 else 0
    fitnessCost = 1/totCost if totCost != 0 else float('inf')


    return maxCost, minCost, averageCost, totCost, fitnessCost

# OGGETTO BIN: rappresentazione dei bin attraverso un oggetto che possiede come
# proprietà la lista degli elementi contenuti e lo spazio occupato
class Bin:
    '''
    Classe rappresentante un bin nel problema del bin-packing.

    Attributi:
        - elementList: lista di booleani che rappresentano la presenza/assenza
        di un determinato elemento nel bin
        - size: spazio occupato del bin.
    
    Metodi:
        - __init__: costruttore della classe.
    '''

    elementList: list[bool]
    size: int | float = 0

    # Costruttore di inizializzazione della classe
    def __init__(self, elementList: list[bool], size: int | float) -> None:
        self.elementList = elementList
        self.size = size

# GREEDY ALGORITHM
# Definisco una funzione con l'approccio greedy in cui viene creata una lista 
# di bin (liste) che si costruisce in maniera tale che ogni bin contenga 
# files per un dimensione totale minore della capacità massima del bin.
# Per costruire tali bin, ogni file viene preso dalla lista generale ed
# inserito all'interno del bin fino a quando non viene raggiunta la capacità massima.
def greedyAlgorithmFiles( setFiles: list[any], binSize: float ) -> list[Bin]:
    '''
    Funzione che sfrutta l'approccio greedy per minimizzare il numero di bin
    contenenti files per una dimensione totale minore della capacità.

    Input:
        - setFiles: lista contenente tutte le dimensioni a disposizione.
        - capacity: intero che rappresenta la dimensione massima del supporto. 

    Output:
        - lista contenente tutti i bin creati con rispettiva dimensione
    '''
    # Inizializzazione della lista contenente i bins creati
    listBins = []

    # Iterazione sopra la lista degli elementi tramite indice ed elemento
    for index, el in enumerate(setFiles):
        
        # Verifica della lista generale se vuota + 
        # verifica dell'elemento corrente se è possibile aggiungerlo alla lista
        # attraverso la verifica dello spazio rimanente all'interno del bin
        if not listBins or all(el.cost > (binSize - bin.size) for bin in listBins):
            
            # Creazione di un nuovo bin contenente l'elemento analizzato
            # che non è possibile aggiungerlo in un bin già esistente
            newBin = [
                False if i != index else True for i in range(len(setFiles))
            ]
            listBins.append(Bin(newBin, el.cost))
        
        else:

            # Ordinamento dei bin in ordine decrescente
            # rispetto alla dimensione occupata
            listBins = sorted(listBins, key=lambda x: x.size, reverse=True)

            # Scorro i bin esistenti e verifico in quale di essi 
            # l'elemento corrente può essere inserito
            for bin in listBins:
                if binSize - bin.size >= el.cost:
                    # Inserisco l'elemento all'interno del bin
                    # + aggiorno la dimensione del bin
                    bin.elementList[index] = True
                    bin.size = bin.size + el.cost
                    break
                else:
                    continue
    
    return listBins

# FUNZIONE STAMPA BIN
def visualizzaBins( bins: list[any]):
    # Conteggio del numero totale di elementi contenuti nei bins
    num = [
        i.elementList.count(True) for i in bins
    ]
    # Liste contenenti gli indici degli elementi di ogni bin
    ind = [
        [
            i for i,v in enumerate(b.elementList) if v == True
        ]
        for b in bins
    ]

    for i in range(len(bins)):
        print(num[i])
        # print(ind[i])

# FUNZIONE DI ENERGY
# Si valuta la qualità della soluzione sulla base delle percentuali di 
# riempimento dei bin attualmente creati: dimensione occupata/dimensione totale.
# Si considera la somma delle percentuali di riempimento: una percentuale 
# che risulta superiore ad 1 viene sottratta in maniera tale da penalizzare 
# i bin sovra-riempiti. 
# Si calcola il numero atteso dei bin totali ed ottenendo una qualità pari al 
# rapporto tra la somma delle percentuali di riempimento ed il numero atteso di bin.
def qualityBin( setBins: list[Bin], binSize: float ) -> float:

    # Calcolo delle percentuali di riempimento di ogni bin creato
    percentuali = [
        bin.size/binSize for bin in setBins
    ]

    # Calcolo della somma delle percentuali di riempimento, penalizzando
    # i bin sovra-riempiti
    sommaPercentuali = 0
    for p in percentuali:
        if p < 1:
            sommaPercentuali += p
        else:
            sommaPercentuali -= p
    
    # Calcolo del numero atteso di bin 
    numAttesoBin = math.ceil(sum(bin.size for bin in setBins)/binSize)

    # Calcolo della qualità della soluzione con i bin attuali
    quality = sommaPercentuali/numAttesoBin

    return quality

# TWEAK OPERATOR
# Funzione di modifica della soluzione attuale definito in maniera randomica
# dove preso randomicamente un elemento all'interno di un bin viene 
# spostato all'interno di un altro bin scelto anch'esso in maniera randomica
def tweakBin( setBins: list[Bin], lenPopulation: int ) -> list[Bin]:

    newSetBins = setBins.copy()

    # Genero tre indici randomici: 
    #   - due indici per selezionare i bin da modificare
    #   - uno per selezionare l'elemento da spostare
    numBins = random.sample(range(len(newSetBins)), 2)
    index = random.randint(0, lenPopulation-1)

    # Estraggo le liste degli elementi dei bin corrispondenti
    fromBin = newSetBins[numBins[0]].elementList
    toBin = newSetBins[numBins[1]].elementList

    # Scambio gli elementi all'interno del bin
    if fromBin[index] and not toBin[index]:
        fromBin[index] = False
        toBin[index] = True
    elif not fromBin[index] and toBin[index]:
        fromBin[index] = True
        toBin[index] = False
    else:
        # Trova un elemento valido da scambiare
        for i in range(len(fromBin)):
            if fromBin[i] and not toBin[i]:
                fromBin[i] = False
                toBin[i] = True
                break
            elif not fromBin[i] and toBin[i]:
                fromBin[i] = True
                toBin[i] = False
                break
    
    return newSetBins

# SIMULATED ANNEALING ALGORITHM
def simulatedAnnealingFiles(
        initialSolution: list[Bin],
        initialTemperature: float,
        minTemperature: float,
        numPopulation: int,
        capacity: float,
        alpha: float
) -> list[Bin]:
    '''
    Funzione che applica l'algoritmo di simulated annealing utilizzando
    una cooling schedule di tipo statico-dinamico.

    Input:
        - initialSolution: soluzione iniziale
        - initialTemperature: temperatura iniziale
        - minTemperature: temperatura minima finale
        - numPopulation: numero totale di individui 
        - capacity: dimensione dei bin
        - alpha: velocità di raffreddamento
    '''

    # Calcolo del numero di steps della cooling schedule utilizzando
    # il parametro tau in funzione del parametro alpha
    tau = (-1)/math.log(alpha)
    L = int(5*tau)

    # Inizializzazione dei parametri
    currentSolution = initialSolution
    temp = initialTemperature

    # Decrescita lenta della temperatura fino alla temperatura minima stabilita
    while temp > minTemperature:

        # Thermal balance test
        for _ in range(L):

            # Modifica della soluzione corrente utilizzando il tweak operator
            copySolution = currentSolution.copy()
            candidateSolution = tweakBin(copySolution, numPopulation)

            # Calcolo del valore del delta: differenza della qualità tra 
            # la soluzione attuale e quella modificata
            delta = qualityBin(candidateSolution, capacity) - qualityBin(currentSolution, capacity)

            # Determinare se la soluzione candidata risulti accettabile 
            if delta < 0:
                # Soluzione accettabile
                solutionSet = candidateSolution
                currentSolution = candidateSolution
            else:
                num = random.uniform(0,1)
                pDelta = math.exp(-delta/temp)
                if num < pDelta:
                    # Soluzione accettata
                    solutionSet = candidateSolution
                    currentSolution = candidateSolution
        
        temp *= alpha

    return solutionSet

# Funzione di FITNESS per la valutazione di un singolo bin
def fitnessFiles(
        population: list[list[bin]],
        binSize: float
) -> list[tuple[list[Bin], float]]:
    
    fittedPopulation = [(ind, qualityBin(ind, binSize)) for ind in population]
    
    return sorted(fittedPopulation, key=lambda x: x[1], reverse=True)

# Funzione di isIdeal per determinare se abbiamo raggiunto la soluzione ideale
def isIdealFiles(
        solution: list[Bin],
        binSize: float,
        targetFitness: float
) -> bool:
    
    return qualityBin(solution, binSize) >= targetFitness

# Funzione di ONE-POINT-CROSSOVER
def crossoverFiles(
        parentA: list[Bin], 
        parentB: list[Bin]
) -> tuple[list[Bin], list[Bin]]:
    
    # Implementa un crossover a punto singolo per i bin
    crossoverPoint = random.randint(0, len(parentA) - 1)
    childA = parentA[:crossoverPoint] + parentB[crossoverPoint:]
    childB = parentB[:crossoverPoint] + parentA[crossoverPoint:]
    
    return childA, childB

# Funzione di MUTAZIONE
def mutationFiles(
        individual: list[Bin], 
        mutationRate: float
) -> list[Bin]:
    
    if random.random() < mutationRate:
        return tweakBin(individual, len(individual))
    
    return individual

# Funzione di SELEZIONE degli individui della popolazione
def selectionFiles(
        population: list[tuple[list[Bin], float]], 
        tournamentSize: int
) -> list[Bin]:
    
    tournament = random.sample(population, tournamentSize)
    tournament = sorted(tournament, key=lambda x: x[1], reverse=True)
    
    return tournament[0][0]


# GENETIC ALGORITHM
def geneticAlgorithmFiles(
        sizePopulation: int,
        numGenerations: int,
        population: list[list[Bin]],
        binSize: float,
        tournamentSize: int = 2,
        targetFitness: float = 0.9,
        mutationRate: float = 0.01,
        fitness: callable = fitnessFiles,
        isIdeal: callable = isIdealFiles,
        crossover: callable = crossoverFiles,
        mutation: callable = mutationFiles,
        selection: callable = mutationFiles
) -> list[Bin]:
    '''
    Funzione che implementa l'algoritmo genetico.

    Input:
        - sizePopulation: dimensione della popolazione desiderata
        - numGenerations: numero di generazioni desiderate
        - population: popolazione di partenza
        - binSize: dimensione massima del bin
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

        fittedPopulation = fitness(population, binSize)

        if best is None:
            best = fittedPopulation[0][0]
        
        for ind in fittedPopulation:
            if ind[1] > qualityBin(best, binSize):
                best = ind[0]
        
        if isIdeal(best, binSize, targetFitness):
            break
        
        nextPopulation = []

        for _ in range(sizePopulation // 2):
            parentA = selection(fittedPopulation, tournamentSize)
            parentB = selection(fittedPopulation, tournamentSize)

            childA, childB = crossover(parentA, parentB)
            nextPopulation.append(mutation(childA, mutationRate))
            nextPopulation.append(mutation(childB, mutationRate))
        
        population = nextPopulation
    
    return best




if __name__ == '__main__':

    #* LETTURA E RACCOLTA DATI
    path = r"C:\Users\palaz\OneDrive\Desktop\University\UNIPA 2.0\II ANNO\Semestre 1\MICO - Machine Intelligence for Combinatorial Optimisation\MICO_coding\Data\Compito_10_06\dataset.txt"
    with open(path, mode="r") as dataset:
        lines = dataset.readlines()
    
    # Pulizia dei dati estrapolati
    # Divisione dei contenuti tra nome file e dimensione in lista
    lines = [
        l.strip().split(',', 1) for l in lines
    ]
    
    # Estrapolazione della capacità dei supporti ottici
    K = float(lines[-1][0].split('=')[1].strip())

    # Estrapolazione del float per la dimensione
    lines = [
        [int(l[0][5:]), float(l[1].split('=')[1].strip())] for l in lines[:-1]
    ]

    # Filtro i valori determinati per rimuovere eventuali valori nan
    linesNan = np.array([
        l[1] for l in lines
    ])
    posNan = np.isnan(linesNan)
    # print(np.where(posNan)) # RESTITUISCE 846, 1106
    # Rimuovo i file con dimensione nan
    lines.pop(1106)
    lines.pop(846)


    # Utilizzo di una namedtuple per immagazzinare tutte informazioni:
    # nome e costo(dimensione) di ogni file considerato
    File = namedtuple( 'File', ['nome', 'cost'])
    files = [
        File(l[0], l[1]) for l in lines
    ]

    maxCost, minCost, meanCost, totCost, fitnessCost = dataAnalysis(files)

    print("---- ORDINE FILES ----")
    print("ANALISI DEI DATI")
    print(f"Dimensione massima: {maxCost}")
    print(f"Dimensione minima: {minCost}")
    print(f"Dimensione media: {meanCost}")
    print(f"Totale somma dimensione: {totCost}")
    print(f"Fitness delle dimensioni: {fitnessCost}")
    print()
    print("Considerazioni:")
    print(f"Sulla base delle analisi, il numero ideale di supporti ottici da utilizzare è pari a {math.ceil(totCost/K)}")
    print("----------------------")

    #* GREEDY APPROACH SOLUTION
    print("-- GREEDY ALGORITHM --")
    greedySolution = greedyAlgorithmFiles(files, K)
    print(f"Numero di bin di dimensione {K} creati = {len(greedySolution)}")
    # print("Bin creati:")
    # visualizzaBins(greedySolution)        
    print("----------------------")

    #* SIMULATED ANNEALING SOLUTION
    deltaE = len(greedySolution)
    t0 = 20*deltaE
    tMin = deltaE/10

    print("- SIMULATED ANNEALING -")
    simulatedSolution = simulatedAnnealingFiles(
        initialSolution=greedySolution,
        initialTemperature=t0,
        minTemperature=tMin,
        numPopulation=len(files),
        capacity=K,
        alpha=0.99
    )
    print(f"Numero di bin di dimensione {K} creati = {len(simulatedSolution)}")
    # print("Bin creati:")
    # visualizzaBins(simulatedSolution)        
    print("----------------------")

    #* GENETIC ALGORITHM SOLUTION
    sizePopulation = 50
    numGenerations = 100
    binSize = 4707319808.0  # Capacità massima del bin
    targetFitness = 0.95  # Fitness target desiderato
    mutationRate = 0.01
    tournamentSize = 5
    initialPopulation = [
        greedyAlgorithmFiles(files, K) for _ in range(sizePopulation)
    ]

    print("-- GENETIC ALGORITHM --")
    geneticSolution = geneticAlgorithmFiles(
        sizePopulation,
        numGenerations,
        initialPopulation,
        binSize,
        tournamentSize,
        targetFitness,
        mutationRate
    )
    print(f"Numero di bin di dimensione {K} creati = {len(geneticSolution)}")
    # print("Bin creati:")
    # visualizzaBins(geneticSolution)        
    print("----------------------")

    print("ANALISI FINALE")
    print(f"Qualità soluzione Greedy: {qualityBin(greedySolution, K)}")
    print(f"Qualità soluzione Simulated: {qualityBin(simulatedSolution, K)}")
    print(f"Qualità soluzione Genetic: {qualityBin(geneticSolution, K)}")
