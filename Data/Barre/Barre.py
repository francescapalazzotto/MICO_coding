'''
    ORDINE BARRE
    Una industria di carpenteria deve fare una lavorazione per cui servono
    pezzi di barre di lunghezza diversa. Le barre devono essere tagliate da
    barre di lunghezza fissa 2000mm.

    Calcolare il numero massimo di barre da 2000mm da comperare.
'''

from collections import namedtuple

# ANALISI DATI: funzione che determina le caratteristiche principali 
# del dataset in osservazione in maniera tale da recuperare informazioni
def dataAnalysis( dataset: type[tuple[any, ...]] ) -> tuple:
    '''
    Funzione progettata per fare un'analisi dei dati a disposizione.
    Inserire in input il dataset riorganizzato all'interno di una namedtuple
    per analizzare e studiare i dati a disposizione.
 
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

# SCORE FUNCTION
# Definisco una funzione score per calcolare la creazione di una barra da 2000mm
# e verificare se è possibile aggiungerne un'altra oppure crearne una nuova
def scoreBin( bucket: list ) -> bool | int:
    '''
    Funzione per determinare la lunghezza del bin (barra da massimo 2000mm).

    Input:
        - bin: lista contenente pezzi di sbarre.

    Output:
        - bool: restituisce true se il bin corrente è massimo, ossia la somma
        dei pezzi contenuti nel bin è pari a 2000mm.
        - int: restituisce i mm mancanti da dover inserire all'interno del bin
        per raggiungere la lunghezza totale di 2000mm.
    '''

    # Calcolo della lunghezza totale del bin
    lunghezzaTot = sum(bucket)

    # Verifica di tale lunghezza
    if lunghezzaTot == 2000:
        return True
    else:
        return 2000 - lunghezzaTot

# GREEDY ALGORITHM
# Definisco una funzione con l'approccio greedy in cui viene creata una lista 
# di bin all'interno di una lista generale che si costruisce in maniera tale
# che ogni bin contenga pezzi di sbarre per un lunghezza totale di 2000mm.
# Per costruire tali bin, ogni barra presa dalla lista generale viene divisa
# e la lunghezza in eccesso viene rimessa all'interno della lista in maniera 
# tale da sfruttarla per la costruizione di un nuovo bin.
def greedyAlgorithmBarre( setBarre: list ) -> int:
    '''
    Funzione che sfrutta l'approccio greedy per massimizzare il numero di bin
    contenenti pezzi di sbarre per una lunghezza totale di 2000mm richiesta.

    Input:
        - setBarre: lista contenente tutte le barre a disposizione.

    Output:
        - int: lunghezza della lista contenente tutti i bin creati da 
        barre da 2000mm.
    '''

    # Lista contenente tutti bin creati
    greedyBarre = []
    # Estrapolo le lunghezze delle barre
    lunghezze = [ barra.cost for barra in setBarre ]
    # Elementi ancora da visionare ordinati in maniera decrescente
    toSeeBarre = sorted(lunghezze, reverse=True)

    while toSeeBarre:

        # Estraggo la lunghezza della barra ad inizio lista per creare i bin
        barra = toSeeBarre.pop(0)
        # Inizializzo una variabile per verificare se la barra
        # corrente viene allocata all'interno di un bin
        allocato = False

        # Creazione del primo bin nel caso in cui soluzione finale è una lista vuota
        if not greedyBarre:
            greedyBarre.append([barra])
        # Scorro i bin all'interno della soluzione finale per inserire
        # la barra appena estrapolata
        else:
            for bucket in greedyBarre:

                # Verifico se il bin corrente è pieno: in caso contrario
                # inserisco il pezzo mancante dalla barra attuale ed inserendo
                # il pezzo in eccesso all'interno delle barre generali
                scoreBucket = scoreBin(bucket)
                if scoreBucket != True:
                    
                    # Verifico che il pezzo mancante sia estraibile dalla barra corrente
                    differenzaBarra = barra - scoreBucket
                    if differenzaBarra >= 0:
                        # Estraggo dalla barra il pezzo mancante e divido i pezzi
                        bucket.append(scoreBucket)
                        if differenzaBarra > 0:
                            toSeeBarre.append(differenzaBarra)
                        allocato = True
                        break
                
            if not allocato:
                greedyBarre.append([barra])
    
    return len(greedyBarre)



                

        






if __name__ == '__main__':

    #* LETTURA E RACCOLTA DATI
    # Leggo i dati dal file e li raccolgo all'interno di una namedtuple
    # per una migliore organizzazione e lettura del codice
    path = r"C:\Users\palaz\OneDrive\Desktop\University\UNIPA 2.0\II ANNO\Semestre 1\MICO - Machine Intelligence for Combinatorial Optimisation\MICO_coding\Data\Barre\dataset.txt"
    with open(path, mode="r") as dataset:
        lines = dataset.readlines()

    # Pulizia dei dati estrapolati
    lines = [ 
        int(l) for l in lines
    ]

    # Utilizzo di una namedtuple per immagazzinare tutte informazioni:
    # nome e costo di ogni barra considerata
    Barra = namedtuple( 'Barra', ['nome', 'cost'])
    barre = [
        Barra('B'+ str(i), lines[i]) for i in range(len(lines))
    ]

    maxCost, minCost, meanCost, totCost, fitnessCost = dataAnalysis(barre)

    print("---- ORDINE BARRE ----")
    print("ANALISI DEI DATI")
    print(f"Lunghezza massima: {maxCost}")
    print(f"Lunghezza minima: {minCost}")
    print(f"Lunghezza media: {meanCost}")
    print(f"Totale somma lunghezze: {totCost}")
    print(f"Fitness delle lunghesse: {fitnessCost}")
    print("----------------------")


    print("GREEDY ALGORITHM")
    greedySolution = greedyAlgorithmBarre(barre)
    print(f"Numero di barre da 2000mm creati = {greedySolution}")
    print("----------------------")
