import os
import sys
from dbscan import paramTuning
from dbscan import dbscan
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '../../Daniele')
from k_means import plot_histogram
from dbscan import plotClusters
import pandas as pd
import pickle as p

filePath = "../../../Dataset/credit_default_cleaned.csv"
df2 = pd.read_csv(filePath)
#paramTuning(df2)

eps = [0.23, 0.25, 0.28, 0.3, 0.32, 0.35, 0.38, 0.42, 0.46, 0.5, 0.55]
minpts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

if not os.path.exists("Clusters/"):
    os.makedirs("Clusters/")
rootPath = "Clusters/"
i = int(sys.argv[1])
myDBscan, d = dbscan(df2, eps[i], minpts[i])
nameCols = df2.columns
for attributeX in nameCols:
    for attributeY in nameCols:
        semiRootPath = rootPath + "cluster" + str(i) + "/"
        if not os.path.exists(semiRootPath):
            os.makedirs(semiRootPath)
        plotClusters(myDBscan, df2, attributeX, attributeY,
                     semiRootPath + attributeX + "_" + attributeY + ".pdf")
print("Finito cluster ", i)
p.dump(d, open(semiRootPath + str(eps) + ".p", "wb"))
print("Serializzato dizionario di ", eps)
