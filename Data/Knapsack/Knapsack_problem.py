'''
KNAPSACK PROBLEM 
This library is constructed in order to implement and find solutions to the Knapsack problem.
'''

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
    objects = sorted(objects, key = lambda x: x[0])


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
    objects = sorted(objects, key = lambda x: x[0])

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
