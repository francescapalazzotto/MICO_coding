'''
    8 QUEEN PUZZLE
    The eight queens puzzle is the problem of placing eight chess queens on 8x8 chessboard,
    so that no two queens threaten each other.
'''

import random

# Implementing a string to take track of the position of the queens in the board:
# each position of the string represents a queen (the row of the board) and we insert in which column to insert the queen. 
pos = []

# Creating the set A containing all the possible positions: 
# uses a list of couples (row, column)
A = [ ( row, col ) for row in range(1,9) for col in range(1,9) ]

# Dividing this set of positions in 3 sets:
#   - X = set of elements chosen -> it will contained the positions chosen to put into the vector pos
#   - Y = set of elements examined and discarded -> it will contain all the positions corresponding to the rows, columns and diagonals to remove
#   - W = set of elements to examine -> it will contained the elements remaining
X = []
Y = []
W = A

while True:

    # Extract a random cell
    position = W.pop( random.randint(0, len(W)-1) )
    X.append(position)

    # Remove all the cells of the same column and same row
    for el in W:
        if el[1] == position[1]:
            Y.append( W.pop(el) )

        if el[0] == position[0]:
            Y.append ( W.pop(el) )
    
    # Remove all the cells of the diagonals
    for j in range(0, position[1]):
        for i in range(0, position[0]):
            for el in W:
                if el[1] == j and el[0] == i:
                    Y.append( W.pop(el) )
        
        for i in range(position[0] + 1, 9):
            for el in W:
                if el[1] == j and el[0] == i:
                    Y.append( W.pop(el) )
    
    for j in range(position[1] + 1, 9):
        for i in range(0, position[0]):
            for el in W:
                if el[1] == j and el[0] == i:
                    Y.append( W.pop(el) )
        
        for i in range(position[0] + 1, 9):
            for el in W:
                if el[1] == j and el[0] == i:
                    Y.append( W.pop(el) )
    
    # Found all the positions
    if len(X) == 8:
        for el in X:
            pos.append( el[1] )
        
        break

    # Not found the positions:
    #   - the queens are not alla positioned
    #   - all the cells are removed (there is no other cell available)
    # if len(X) < 8 and W 


        