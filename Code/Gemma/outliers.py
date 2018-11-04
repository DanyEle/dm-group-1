import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
"""Questa funzione individua gli outlier in una determinata colonna colName del file filePath.
Osservazione da Wikipedia: in statistica viene definito outlier un valore al di fuori dall'intervallo: [Q1 - k(Q3-Q1), Q3 + k(Q3-Q1)] """


def countOutliers(dataFrame, colName, k):
    currCol = getattr(dataFrame, colName)
    count = 0
    Q1 = np.percentile(currCol, 25)
    Q3 = np.percentile(currCol, 75)
    lower = Q1 - k * (Q3 - Q1)
    upper = Q3 + k * (Q3 - Q1)
    for i in range(0, len(currCol)):
        if not ((currCol[i] > lower) & (currCol[i] < upper)):
            count = count + 1
    return count


def removeOutliers(dataFrame):
    baMay = getattr(dataFrame, "ba-may")
    baApr = getattr(dataFrame, "ba-apr")
    paAug = getattr(dataFrame, "pa-aug")
    rows = []
    for i in range(0, len(baMay)):
        if ((int(baMay[i]) < -5000) | (int(baApr[i]) < -5000) |
            (int(paAug[i]) > 1200000)):
            rows.append(i)
    print("Number of rows to be dropped: ", len(rows))
    dataFrame.drop(dataFrame.index[rows], inplace=True)
    print("size: ", len(dataFrame))
    return dataFrame
