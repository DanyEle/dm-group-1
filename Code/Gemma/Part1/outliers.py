import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
"""Questa funzione individua gli outlier in una determinata colonna colName del data frame dataFrame.
Osservazione da Wikipedia: in statistica viene definito outlier un valore al di fuori dell'intervallo: [Q1 - k(Q3-Q1), Q3 + k(Q3-Q1)] """


def countOutliers(dataFrame, colName, k):
    currCol = getattr(dataFrame, colName)
    count = 0
    Q1 = np.percentile(currCol, 25)
    Q3 = np.percentile(currCol, 75)
    lower = Q1 - k * (Q3 - Q1)
    upper = Q3 + k * (Q3 - Q1)
    for i in range(0, len(currCol)):
        if not ((int(currCol.iloc[i] > lower)) and
                (int(currCol.iloc[i] < upper))):
            count = count + 1
    return count


"""Questa funzione elimina gli outlier individuati tramite una visual analysis in place nel data frame dataFrame"""


def removeOutliers(dataFrame):
    print("Initial size of data frame: ", dataFrame.shape)
    baMay = getattr(dataFrame, "ba-may")
    baApr = getattr(dataFrame, "ba-apr")
    paAug = getattr(dataFrame, "pa-aug")
    paApr = getattr(dataFrame, "pa-apr")
    paMay = getattr(dataFrame, "pa-may")
    rows = []
    for i in range(0, len(baMay)):
        if ((int(baMay[i]) < -5000) | (int(baApr[i]) < -5000) |
            (int(paAug[i]) > 500000) | (int(paApr[i]) > 500000) |
            (int(paMay[i]) > 400000)):
            rows.append(i)
    print("Visual analysis, number of rows to be dropped: ", len(rows))
    dataFrame.drop(dataFrame.index[rows], inplace=True)
    print("Final size of data frame: ", dataFrame.shape)
    return
