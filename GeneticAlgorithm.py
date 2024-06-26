'''
    GENETIC ALGORITHM
'''
import random

def fitness( individuo: any ) -> float:
    '''
    Funzione di fitness per la valutazione degli individui
    sulla base della tipologia del problema analizzato.
    '''
    return None

#* SELECTION OPERATORS
# Funzioni per la selezione degli individui che genereranno altri figli 
def tournamentSelection(
        population: list[any],
        fitness: callable,
        t: int = 2
) -> list[any]:
    '''
    Effettua la torunament selection su una popolazione.

    Input:
        - population: lista di vettori/liste rappresentanti la popolazione
        - fitness: funzione per il calcolo del valore di fitness di un vettore
        - t: dimensione del tournament/numero di partecipanti, default 2.
    
    Output:
        - vettore/lista selezionato
    '''
    best = None
    for _ in range(t):
        nextCandidate = population[random.randint(0, len(population)-1)]
        
        if not best or fitness(nextCandidate) > fitness(best):
            best = nextCandidate
    
    return best

def tournamentSelectionExtraction(
        population: list[any],
        fitness: callable,
        t: int = 2
) -> list[any]:
    '''
    Effettua la torunament selection su una popolazione con estrazione.

    Input:
        - population: lista di vettori/liste rappresentanti la popolazione
        - fitness: funzione per il calcolo del valore di fitness di un vettore
        - t: dimensione del tournament/numero di partecipanti, default 2.
    
    Output:
        - vettore/lista selezionato
    '''
    best = None
    for _ in range(t):
        nextCandidate = population.pop(random.randint(0, len(population)-1))
        
        if not best or fitness(nextCandidate) > fitness(best):
            best = nextCandidate
    
    return best

#* CROSSOVER FUNCTIONS
def onePointCrossover( 
        v: list[any], 
        w: list[any], 
        c: int 
) -> tuple[list[any], list[any]]:
    '''
    Applicazione del one-point crossover su due vettori nel punto c indicato. 

    Input:
        - v: primo vettore
        - w: secondo vettore
        - c: punto di crossover
    
    Output:
        - tupla contenente i due nuovi vettori dopo la trasformazione.
    '''

    return v[:c]+w[c:], w[:c]+v[c:]

def nPointsCrossover(
        v: list[any], 
        w: list[any], 
        l: list[int]    
) -> tuple[list[any], list[any]]:
    '''
    Applicazione del n-points crossover su due vettori nei punti riportarti
    all'interno della lista l. 

    Input:
        - v: primo vettore
        - w: secondo vettore
        - l: punti di crossover
    
    Output:
        - tupla contenente i due nuovi vettori dopo la trasformazione.
    '''

    if len(l) == 1:
        return onePointCrossover(v,w,l[0])
    else:
        i,j = nPointsCrossover( w[l[0:]], v[l[0]:], l[1:])
        return v[:l[0]]+i, w[:l[0]]+j

def randomShuffle(v: list[any] ) -> list[any]:
    '''
    Mescola randomicamente gli elementi di un vettore/lista.
    '''
    return random.shuffle(v)


#* ALGORITMO GENETICO
def geneticAlgorithm(
        sizePopulation: int,
        numGenerations: int,
        population: list[any],
        fitness: callable,
        isIdeal: callable,
        crossover: callable,
        mutation: callable,
        selection: callable
) -> list[any]:
    '''
    Funzione che implementa l'algoritmo genetico.

    Input:
        - sizePopulation: dimensione della popolazione desiderata
        - numGenerations: numero di generazioni desiderate
        - population: popolazione di partenza
        - fitness: funzione di valutazione di ogni individuo
        - isIdeal: funzione di valutazione della popolazione totale
        - crossover: funzione per l'applicazione del crossover
        - mutation: funzione per la mutazione degli individui
        - selection: funzione per la selezione degli individui
    
    Output:
        - lista degli individui ottenuti
    '''

    # Inizializzazione della prima generazione al tempo 0 + variabile per
    # la migliore soluzione trovata
    time = 0
    best = None

    # Implementazione del ciclo per le generazioni desiderate + valutazione
    # della best soluzione trovata
    #* VALUTARE LA CREAZIONE DI isIdeal SULLA BASE DELL'OBIETTIVO DESIDERATO
    while not isIdeal(best) and time != numGenerations:

        # Valutazione di ogni cromosoma (individuo - possibile soluzione) 
        # della popolazione sulla base della fitness function creata
        #* Funzione possibilmente che valuta la fitness di ogni individuo
        #* e riordina la lista delle soluzione in ordine decrescente
        #* Lista di tuple: lista della soluzione + valore di fitness corrispondente
        fittedPopulation = fitness(population)

        if best == None:
            best = population[0]
        for ind in fittedPopulation:
            # Scorro tutte le soluzioni e controllo se esiste una soluzione
            # con fitness maggiore della migliore trovata: se si, sovrascrivo
            if ind[1] > fitness(best):
                best = ind[0]

        # Inizializzazione della popolazione alla generazione successiva     
        nextPopulation = []

        for _ in range(1, (sizePopulation//2)):

            # Selezione dei genitori per la riproduzione di nuovi individui
            parentA = selection(population)
            parentB = selection(population)

            # Step di crossover per la generazione dei figli
            childA, childB = crossover(parentA, parentB)
            nextPopulation.append(mutation(childA))
            nextPopulation.append(mutation(childB))
        
        population = nextPopulation
        time += 1
    
    return best