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

# DATA ANALYSIS: sfruttare libreria namedtuple
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

# Funzione di score: ogni bin è rappresentato da liste binarie
# indicanti quali file vengono presi in considerazione
def scoreFiles( individual: list, dataset: type[tuple[any, ...]] ) -> int:
    '''
    Funzione che assegna uno score ad un individuo sulla base della funzione
    obiettivo: verifica la dimensione massima del bin come somma delle dimensioni
    di ogni file in esso contenuto.

    Input: 
        - individual: bin attuale rappresentato da una lista binaria
        - dataset: set dei dati generale per accedere alle dimensione
        dei file considerati all'interno della soluzione

    Output:
        - int: valore totale della dimensione del bin analizzato
    '''

    individual = np.array(individual)
    # Estrapolo gli indici delle posizioni in cui il valore è pari ad 1
    # per accedere alle informazioni delle dimensioni del file associato
    indexes = np.where(individual == 1)[0]
    
    # Calcolo dello score del bin come somma delle dimensioni di ogni file
    score = 0
    for pos in indexes:
        score += dataset[pos].cost
    
    return score

# GREEDY ALGORITHM
# Definisco una funzione con l'approccio greedy in cui viene creata una lista 
# di bin (liste) che si costruisce in maniera tale che ogni bin contenga 
# files per un dimensione totale minore della capacità massima del bin.
# Per costruire tali bin, ogni file viene preso dalla lista generale ed
# inserito all'interno del bin fino a quando non viene raggiunta la capacità massima.
def greedyAlgorithmFiles( setFiles: type[tuple[any, ...]], capacity: float ) -> list:
    '''
    Funzione che sfrutta l'approccio greedy per minimizzare il numero di bin
    contenenti files per una dimensione totale minore della capacità.

    Input:
        - setFiles: lista contenente tutte le dimensioni a disposizione.
        - capacity: intero che rappresenta la dimensione massima del supporto. 

    Output:
        - lista contenente tutti i bin creati con rispettiva dimensione
    '''

    # Lista contenente tutti bin creati
    greedyFiles = []
    # Elementi ancora da visionare ordinati in maniera decrescente
    toSeeFiles = sorted(setFiles.copy(), key=lambda x: x.cost, reverse=True)

    # Itero sopra la lista dei file da visionare andando a creare i bin
    # considerando i file in ordine decrescente fino a quando non raggiungo
    # la capacità massima disponibile, fino a quando la lista non diventa vuota
    while toSeeFiles:
        # Itero sopra la lista bloccando il ciclo una volta raggiunta la capacità
        
        # Variabile utilizza per verificare la dimensione del bin attuale
        dimBin = 0

        # Variabile per creare il bin da inserire nella soluzione
        bucket = []

        for file in toSeeFiles:
            # Se l'aggiunta del file con la sua dimensione rientra nella capacità
            # inserisco il file nel bin desiderato
            dimBin += file.cost
            if dimBin <= capacity:
                bucket.append(file)
                toSeeFiles.remove(file)
            else:
                break
        
        greedyFiles.append(bucket)
    
    return greedyFiles

# Funzione definita per modificare in maniera randomica il bin considerato
def tweakOperator( setBins: list, individuals: list, capacity: float) -> list:
    '''
    Funzione per modificare la soluzione corrente randomicamente e valutando
    la modifica in base al valore dello score. La modifica avviene 
    spostando randomicamente dei file nei vari bucket e verificando
    sempre la capacità massima del bin possibile.

    Input: 
        - setBins: soluzione attuale contenente tutti i bins creati.
        - individuals: lista di tutti gli individui per accedere al loro costo.
        - capacity: capacità massima da rispettare per il bin considerato

    Output:
        - lista binaria rappresentatnte la soluzione modificata.
    '''

    while True:

        # Faccio muovere randomicamente 2 files per ogni bin
        nMosse = 2

        # Creazione nuova lista di bins effettuando una copia dell'originale
        newSetBins = [
            bucket.copy() for bucket in setBins
        ]

        # Generazione delle mosse randomiche tra i vari bin
        for _ in range(nMosse):
            # Seleziono un bin randomicamente
            binIndex = random.randrange(len(setBins))
            binModified = newSetBins[binIndex]

            # Verifico che il bin non sia vuoto
            if binModified.count(1) > 0:
                # Seleziono un file al suo interno in maniera randomica
                fileIndex = random.choice([i for i, val in enumerate(binModified) if val == 1])
                fileValue = binModified[fileIndex]

                # Rimuovo il file spostato dal bin selezionato
                binModified[fileIndex] = 0

                # Seleziono una destinazione randomicamente 
                destBinIndex =  random.choice([i for i in range(len(setBins)) if i != binIndex])
                destBin = newSetBins[destBinIndex]

                # Aggiungo il nuovo file nel bin di destinazione
                destBin[fileIndex] = fileValue
        
        # Valuto lo score dei nuovi bin creati attraverso la funzione di score
        # Se tutti i bin creati hanno dimensione inferiore alla capacità massima
        # ritorna il nuovo set di bins creati
        check = []
        for bucket in newSetBins:
            if scoreFiles(bucket, individuals) <= capacity:
                check.append(True)
            else:
                check.append(False)
        
        # Verifico che il check sia composto da solo componenti True
        if False not in check:
            break
    
    return newSetBins

# Funzione per valutare l'energia e quindi il numer totale di bin creati una volta
# modificati i bins generati
def energyFunction( xSolution: list ) -> int:
    '''
    Funzione che conta il numero di bin contenuti nella soluzione.
    '''
    return len(xSolution)

# SIMULATED ANNEALING
def simulatedAnnealingFiles( 
        
        initial_solution : list,
        initial_temperature : float,
        individuals: type[tuple[any, ...]],
        tMin: float,
        capacity: float,
        alpha : float 

    ) -> list:
    
    '''
    Function that applies the simulated annealing function
    using an hybrid cooling schedule (static-dynamic).

    Input:
        - initial solution
        - initial temperature: a high number (tends to 0)
        - individuals: set of all files
        - tMin: minimum temperature
        - alpha: float number less than 1, velocity of the cooling schedule
    
    Output:
        - solutions set: list containing all the solution chosen
    '''

    # Calculating the number of steps of the cooling schedule
    # using the parameter tau in function of alpha
    tau = (-1) / math.log(alpha)
    L = int(5 * tau)
    
    # Initialization of the parameters
    sol_current = initial_solution
    temp = initial_temperature

    solutions_set = [] # It could be a single solution
    # Temperature decreases slowly until 0
    while temp > tMin:

        # Thermal balance test
        for _ in range(L):
            
            # Tweaking the current solution
            sol_copy = sol_current.copy()
            sol_candidate = tweakOperator(sol_copy, individuals, capacity)

            # Calculate the value of delta: difference between the energies
            # of the current solution and the modified one
            delta = energyFunction(sol_candidate) - energyFunction(sol_current)

            # Determine if the candidate solution is acceptable
            if delta < 0:

                # Accepted
                solutions_set.append(sol_candidate)
                sol_current = sol_candidate

            else:

                num = random.uniform(0,1) # INSERIRE CONDIZIONE CHE MI AIUTI A NON PRENDERE 0
                p_delta = math.exp( -delta/ temp )
                if num < p_delta:

                    # Accept the solution
                    solutions_set.append(sol_candidate)
                    sol_current = sol_candidate

        # Cooling schedule: static
        temp *= alpha 

    return solutions_set 






if __name__ == '__main__':

    #* LETTURA E RACCOLTA DATI
    path = r"C:\Users\palaz\OneDrive\Desktop\University\UNIPA 2.0\II ANNO\Semestre 1\MICO - Machine Intelligence for Combinatorial Optimisation\MICO_coding\Data\Compito_10_06\dataset.txt"
    with open(path, mode="r") as dataset:
        lines = dataset.readlines()
    
   

    # Pulizia dei dati estrapolati
    # Dvisione dei contenuti tra nome file e dimensione in lista
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
    print("GREEDY ALGORITHM")
    greedySolution = greedyAlgorithmFiles(files, K)
    print(f"Numero di bin di dimensione {K} creati = {len(greedySolution)}")
    print("Bin creati:")
    for bucket in greedySolution:
        print(bucket)
    print("----------------------")
    
    # Trasformazione della soluzione greedy in lista di individui binari
    greedyBin = []
    for bucket in greedySolution:
        indexes = [
            el.nome-1 for el in bucket
        ]
        binBucket = [0]*len(files)
        for ind in indexes:
            if ind > 846 or ind > 1106:
                ind -= 1
            binBucket[ind-1] = 1
        greedyBin.append(binBucket)

    #* SIMULATED ANNEALING ALGORITHM
    deltaE = len(greedyBin)
    T0 = 20*deltaE
    Tmin = deltaE/10

    print("SIMULATED ANNEALING")
    simulatedSolution = simulatedAnnealingFiles(
        initial_solution=greedyBin,
        initial_temperature=T0,
        tMin=Tmin,
        individuals=files,
        capacity=K,
        alpha=0.99
    )
    print(f"Numero di bin di dimensione {K} creati = {len(simulatedSolution)}")
    print("----------------------")