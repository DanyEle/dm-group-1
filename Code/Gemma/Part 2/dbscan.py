import math
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

filePath = "../../Dataset/credit_default_train.csv"

dataFrame = pd.read_csv(filePath)
deep1 = dataFrame.copy()
deep2 = dataFrame.copy()
dataFrameNoMV = remove_missing_values(deep1)
df1 = removeOutliers(deep2)

scaler = MinMaxScaler()  #normalization
df1 = scaler.fit_transform(df2.values)  #df1 is normalized
matrix = pd.DataFrame(squareform(pdist(df1,
                                       metric="euclidean")))  #distance matrix
matrixSorted = matrix[4000].sort_values()  #sorted dinstances to choose epsilon

plt.plot(matrixSorted.tolist())
