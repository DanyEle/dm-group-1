"""Questa funzione ritorna l'indice dell'elemento che più si avvicina alla media del vettore array"""


def findMeanIndex(array):
    if (len(array) % 2 == 0):
        mean = sum(array) / len(array)
        meanIndexL = len(array) // 2
        meanIndexU = len(array) // 2 + 1
        if (mean - array[meanIndexU] >= mean - array[meanIndexL]):
            meanIndex = meanIndexL
        else:
            meanIndex = meanIndexU
    else:
        meanIndex = len(array) / 2
    return int(meanIndex)


def findMedian(array):
    if (len(array) % 2 == 0):
        pos = int(len(array) / 2)
        return array[pos], pos
    else:
        left = int(len(array) // 2)
        right = int(left + 1)
        pos = (array[left] + array[right]) / 2
        return pos, left


"""Questa funzione ritorna il valore dello spread del vettore myCol, a seconda del tipo desiderato (kind). Kind può valere "range", "irange" o "variance" """


def spread(myCol, kind):
    if (kind == "range"):
        mySpread = max(myCol) - min(myCol)
        return mySpread
    if (kind == "irange"):
        myCol.sort()
        meanIndex = findMeanIndex(myCol)
        left = []
        #print("myCol: ", myCol[0])
        left = [myCol[l] for l in range(0, meanIndex)]
        right = [myCol[l] for l in range(meanIndex, len(myCol))]
        leftMean = findMeanIndex(left)
        rightMean = findMeanIndex(right)
        return rightMean - leftMean
    if (kind == "variance"):
        #Find mean
        mean = sum(myCol) / len(myCol)
        #Calculate vector of squares of differences between
        #the mean and the current element
        differences = [(mean - myCol[l]) * (mean - myCol[l])
                       for l in range(0, len(myCol))]
        #Sum all the elements of the vector
        return sum(differences)
