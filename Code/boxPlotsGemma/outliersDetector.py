import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
"""This function takes as an input the path of the file that represents the dataset (filePath) and the string which is the name of a column (colName). This function returns the column as a list or -1, if thecolumn doesn't have numerical values"""


def highlightColumn(filePath, colName, pandas=None):
    if ((colName == "sex") | (colName == "education") | (colName == "status") |
        (colName == "credit_default")):
        print("Cannot print boxplot from non numerical data\n")
        return -1
    if ((colName != "limit") & (colName != "age") & (colName != "ps-sep") &
        (colName != "ps-sep") & (colName != "ps-aug") & (colName != "ps-jul") &
        (colName != "ps-jun") & (colName != "ps-may") & (colName != "ps-apr") &
        (colName != "ba-sep") & (colName != "ba-sep") & (colName != "ba-aug") &
        (colName != "ba-jul") & (colName != "ba-jun") & (colName != "ba-may") &
        (colName != "ba-apr") & (colName != "pa-sep") & (colName != "pa-sep") &
        (colName != "pa-aug") & (colName != "pa-jul") & (colName != "pa-jun") &
        (colName != "pa-may") & (colName != "pa-apr")):
        print("Typo in the column name\n")
        return -1
    else:
        if (pandas):
            data = filePath
            return getattr(data, colName)
        else:
            data = open(filePath, "r")
            if (colName == "limit"):
                col = 0
            if (colName == "age"):
                col = 4
            if (colName == "ps-sep"):
                col = 5
            if (colName == "ps-aug"):
                col = 6
            if (colName == "ps-jul"):
                col = 7
            if (colName == "ps-jun"):
                col = 8
            if (colName == "ps-may"):
                col = 9
            if (colName == "ps-apr"):
                col = 10
            if (colName == "ba-sep"):
                col = 11
            if (colName == "ba-aug"):
                col = 12
            if (colName == "ba-jul"):
                col = 13
            if (colName == "ba-jun"):
                col = 14
            if (colName == "ba-may"):
                col = 15
            if (colName == "ba-apr"):
                col = 16
            if (colName == "pa-sep"):
                col = 17
            if (colName == "pa-aug"):
                col = 18
            if (colName == "pa-jul"):
                col = 19
            if (colName == "pa-jun"):
                col = 20
            if (colName == "pa-may"):
                col = 21
            if (colName == "pa-apr"):
                col = 22
            lines = [list(l.split(',')) for l in data]
            selectedCol = []
            #print("Colonna ", colName)
            for i in range(1, len(lines)):
                #print("colonna vera: ", lines[i][col])
                selectedCol.append(int(lines[i][col]))
                #print(selectedCol[i - 1])
    return selectedCol


def findMedian(array):
    if (len(array) % 2 == 0):
        pos = int(len(array) / 2)
        return array[pos], pos
    else:
        left = int(len(array) // 2)
        right = int(left + 1)
        pos = (array[left] + array[right]) / 2
        return pos, left


def countOutliers(filePath, colName, pandas=None):
    uCurrCol = highlightColumn(filePath, colName, pandas)
    count = 0
    currCol = uCurrCol.sort_values()
    print(currCol)
    (median, pos) = findMedian(currCol)
    print("Median: ", median)
    first = [currCol[i] for i in range(0, pos)]
    third = [currCol[i] for i in range(pos + 1, len(currCol))]
    (firstQuartile, pFQ) = findMedian(first)
    (thirdQuartile, fFQ) = findMedian(third)
    print("Third quartile: ", thirdQuartile)
    print("First quartile: ", firstQuartile)
    print("Third - first: ", thirdQuartile - firstQuartile)

    k = 1.5
    lower = firstQuartile - k * (thirdQuartile - firstQuartile)
    upper = thirdQuartile + k * (thirdQuartile - firstQuartile)
    print("Lower limit: ", lower)
    print("Upper limit: ", upper)
    for i in range(0, len(currCol)):
        if not ((currCol[i] > lower) & (currCol[i] < upper)):
            #            print("Elemento: ", currCol[i])
            count = count + 1
    return i
