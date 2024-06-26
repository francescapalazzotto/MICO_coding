'''
    Funzioni utili da poter utilizzare all'interno dei codici.
'''
import random

# Funzione di score: l'individuo è rappresentato da liste binarie
def score( individual: list ) -> int:
    '''
    Funzione che assegna uno score ad un individuo sulla base della funzione
    obiettivo: in questo esempio, si suppone di contare se ogni componente
    della soluzione copre una determinata mansione. 
    Conta il numero di zeri presenti nella lista binaria che contiene ogni
    mansione. 
    '''

    score = sum( 1 if el == 0 else 0 for el in individual )

    return score

# TWEAK OPERATOR
def tweakOperator( xSolution: list, individuals: list) -> list:
    '''
    Funzione per modificare la soluzione corrente randomicamente e valutando
    la modifica in base al valore dello score. 

    Input: 
        - xSolution: soluzione attuale binaria da modificare.
        - individuals: lista di tutti gli individui per accedere al loro costo.

    Output:
        - lista binaria rappresentatnte la soluzione modificata.
    '''

    while True:

        xTweaked = xSolution.copy()
        
        # Choosing a random index to change in the current solution
        index = random.randint(0, len(xSolution)-1)

        # Changing the corresponding position in the solution
        # If it is 0, change it in 1; and viceversa
        xTweaked[index] = 0 if xTweaked[index] == 1 else 1
    
        # Checking if the new solution generated is acceptable
        #* INFO da sostituire con l'informazione da estrapolare dagli individui
        xTweakedDimensions = [
            individuals[i-1].info for i in range(len(xSolution)) if xTweaked[i] == 1
        ]

        # Evaluate the score of this tweaked solution through the dimensions
        tweakedSolutionScore = score(xTweakedDimensions)
        #* MODIFICA CONDIZIONE PER IL BREAK SULLA BASE DELLA FUNZIONE OBIETTIVO
        if tweakedSolutionScore >= 0:
            break
    
    return xTweaked

# Random solution
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
            1 if i in positions else 0 for i in range(len(individuals)) 
        ]

        randomInfo = [
            individuals[i-1].info for i in range(len(randomSolution)) if randomSolution[i] == 1
        ]

        #* CONDIZIONE PER LA QUALE LO SCORE è ACCETTABILE
        if score(randomInfo) >= 0:
            break
    
    return randomSolution

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

    return max(costRates), min(costRates), sum(costRates)/len(costRates), sum(costRates), 1/sum(costRates)
