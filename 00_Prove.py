from collections import namedtuple
import random

def qualitySpot(
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
Spot = namedtuple( 'Spot', ['nome', 'time', 'cost'])
spots = [
    Spot(1, 120, 100),
    Spot(2, 50, 50),
    Spot(3, 10, 200),
    Spot(4, 100, 10),
]
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
        index = random.randint(0, len(xSolution)-1)

        # Changing the corresponding position in the solution
        # If it is False, change it in True; and viceversa
        xTweaked[index] = False if xTweaked[index] == True else True
    
        # Checking if the new solution generated is acceptable
        xTweakedTime = 0
        for index, el in enumerate(xTweaked):
            print('el', el)
            print('index', index)
            if el:
                xTweakedTime = xTweakedTime + setSpots[index].time
                print('xTweakedTime',xTweakedTime)

        # Evaluate the score of this tweaked solution through the dimensions (time)
        if xTweakedTime <= maxTime:
            break
    
    return xTweaked

greedySolutionBin = [False]*7
for el in spots:
    greedySolutionBin[el.nome] = True

print(greedySolutionBin)
