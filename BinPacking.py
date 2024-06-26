'''
    BIN PACKING PROBLEM

    Logica seguita:
        - i bin vengono segnati come vettori binari [presente, assente]
        - QUALITY: sommiamo percentuali di riempimento minori o uguali a 1
        e sottraiamo percentuali maggiori di 1
            quality = somme()/numero atteso di bin
        - TWEAK: prendiamo un elemento random e lo spostiamo in un bin random
        - ISIDEAL: 
            - 0.9 <= percentuali di riempimento <= 1
            - numero bin = numero di bin atteso
'''

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

# FUNZIONE CREAZIONE DELLA LISTA DEI BINS
# Ogni elemento del bin è rappresentato mediante il suo peso, in base al tipo
# di problema da dover risolvere
def binPacking(
    elements: list[any],
    binSize: float
) -> list[Bin]:
    
    # Inizializzazione della lista contenente i bins creati
    listBins = []

    # Iterazione sopra la lista degli elementi tramite indice ed elemento
    for index, el in enumerate(elements):
        
        # Verifica della lista generale se vuota + 
        # verifica dell'elemento corrente se è possibile aggiungerlo alla lista
        # attraverso la verifica dello spazio rimanente all'interno del bin
        if not listBins or all(el > (binSize - bin.size) for bin in listBins):
            
            # Creazione di un nuovo bin contenente l'elemento analizzato
            # che non è possibile aggiungerlo in un bin già esistente
            newBin = [
                False if i != index else True for i in range(len(elements))
            ]
            listBins.append(Bin(newBin, el))
        
        else:

            # Ordinamento dei bin in ordine decrescente
            # rispetto alla dimensione occupata
            listBins = sorted(listBins, key=lambda x: x.size, reverse=True)

            # Scorro i bin esistenti e verifico in quale di essi 
            # l'elemento corrente può essere inserito
            for bin in listBins:
                if binSize - bin.size >= el:
                    # Inserisco l'elemento all'interno del bin
                    # + aggiorno la dimensione del bin
                    bin.elementList[index] = True
                    bin.size = bin.size + el
                    break
                else:
                    continue
    
    return listBins

# FUNZIONE STAMPA BIN
def visualizzaBins( bins: list[any] ):
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
        print(i, "*"*num[i])
        print(ind[i])
