import os
from outliersDetector import countOutliers
import pandas as pd
import sys
sys.path.insert(0, '../Riccardo')
from MissingValues_2 import remove_missing_values

filePath = "../../Dataset/credit_default_train.csv"
ccOld = pd.read_csv(filePath)
cc = remove_missing_values(ccOld)
totalLength = len(cc.limit)
k = 1.5
print("Lines in dataset: ", totalLength)

limit = countOutliers(cc, "limit", k)
print("Outliers in limit: ", limit)
age = countOutliers(cc, "age", 1)
print("Outliers in age: ", age)

pssep = countOutliers(cc, "ps-sep", k)
print("Outliers in ps-sep: ", pssep)

psaug = countOutliers(cc, "ps-aug", k)
print("Outliers in ps-aug: ", psaug)

psjul = countOutliers(cc, "ps-jul", k)
print("Outliers in ps-jul: ", psjul)

psjun = countOutliers(cc, "ps-jun", k)
print("Outliers in ps-jun: ", psjun)

psmay = countOutliers(cc, "ps-may", k)
print("Outliers in ps-may: ", psmay)

psapr = countOutliers(cc, "ps-apr", k)
print("Outliers in ps-apr: ", psapr)

basep = countOutliers(cc, "ba-sep", k)
print("Outliers in ba-sep: ", basep)

baaug = countOutliers(cc, "ba-aug", k)
print("Outliers in ba-aug: ", baaug)

bajul = countOutliers(cc, "ba-jul", k)
print("Outliers in ba-jul: ", bajul)

bajun = countOutliers(cc, "ba-jun", k)
print("Outliers in ba-jun: ", bajun)

bamay = countOutliers(cc, "ba-may", k)
print("Outliers in ba-may: ", bamay)

baapr = countOutliers(cc, "ba-apr", k)
print("Outliers in ba-apr: ", baapr)

pasep = countOutliers(cc, "pa-sep", k)
print("Outliers in pa-sep: ", pasep)

paaug = countOutliers(cc, "pa-aug", k)
print("Outliers in pa-aug: ", paaug)

pajul = countOutliers(cc, "pa-jul", k)
print("Outliers in pa-jul: ", pajul)

pajun = countOutliers(cc, "pa-jun", k)
print("Outliers in pa-jun: ", pajun)

pamay = countOutliers(cc, "pa-may", k)
print("Outliers in pa-may: ", pamay)

paapr = countOutliers(cc, "pa-apr", k)
print("Outliers in pa-apr: ", paapr)
