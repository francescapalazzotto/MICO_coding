'''
    COMPITO 13 GIUGNO 2017
'''








if __name__ == "__main__":

    # Reading the csv file containing the dimensions of each file
    # that we need to backup and the dimension K of the optical devices used.
    path = r"C:\Users\palaz\OneDrive\Desktop\University\UNIPA 2.0\II ANNO\Semestre 1\MICO - Machine Intelligence for Combinatorial Optimisation\MICO_coding\Data\Backup - Optical\dataset.csv"
    data = open(path)
    files = data.readlines()
    
    # List containing couples of (file, dimension)
    register = []
    for fil in files[:len(files)-3]:
        div = fil.split( sep = "," )

        # Checking if there are missing values and replacing them with 0
        if div[2] == 'nan\n':
            register.append( ( div[0], float(0) ) )
        else:
            register.append( ( div[0], float( (div[2].split())[0] ) ) )
    
    # Capacity of the optical devices
    K = float( (files[len(files)-1].split( sep = "," )[1]).split()[0] )
    print(f"K{K}")
    # Determine the theoretical value of the minimum that we should obtain
    value_t = sum( dim[1] for dim in register )/K
    print(value_t)