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
