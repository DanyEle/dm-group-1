import os
import sys
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as p

sys.path.insert(0, '../../Daniele')
from dbscan import plotEpsMinPts
from dbscan import EpsMinPtsEvaluation

filePath = "../../../Dataset/credit_default_cleaned.csv"
df = pd.read_csv(filePath)
#plotEpsMinPts(df)

eps = [0.23, 0.25, 0.28, 0.3, 0.32, 0.35, 0.38, 0.42, 0.46, 0.5, 0.55]
minpts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

if not os.path.exists("Clusters/"):
    os.makedirs("Clusters/")
rootPath = "Clusters/"

EpsMinPtsEvaluation(eps, minpts, rootPath, df)
