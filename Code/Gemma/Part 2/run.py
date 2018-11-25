import os
import sys
from dbscan import paramTuning
from dbscan import dbscan

import pandas as pd

filePath = "../../../Dataset/credit_default_cleaned.csv"
df2 = pd.read_csv(filePath)
#paramTuning(df2)

#Once we decided the values fo eps and minpts

eps = 0.25
minpts = 32
dbscan(df2, eps, minpts)
