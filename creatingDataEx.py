import numpy as np

def cratingDataSetFromFile(fileName):
    f = open(fileName, encoding='utf8')

    X = []
    Y = []

    for line in f:
        s = np.array([int(x) for x in line.split()])
        x = s[0:3]
        y = s[3]
        if( len(X) == 0 ):
            X = [x/255]
            Y = [y%2]
        else:
            X = np.append(X, [x/255], axis=0)
            Y = np.append(Y, [y%2], axis=0)

    print(X)
    print(Y)
    np.save('outputData', Y)
    np.save('inputData', X)

    print("Saved data!")

cratingDataSetFromFile(fileName)