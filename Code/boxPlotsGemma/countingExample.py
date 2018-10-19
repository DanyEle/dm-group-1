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

print("Lines in dataset: ", totalLength)

limit = countOutliers(cc, "limit", 1)
print("Outliers in limit: ", limit)
'''age = countOutliers(cc, "age", 1)
print("Outliers in age: ", age)

pssep = countOutliers(cc, "ps-sep", 1)
print("Outliers in ps-sep: ", pssep)

psaug = countOutliers(cc, "ps-aug", 1)
print("Outliers in ps-aug: ", psaug)

psjul = countOutliers(cc, "ps-jul", 1)
print("Outliers in ps-jul: ", psjul)

psjun = countOutliers(cc, "ps-jun", 1)
print("Outliers in ps-jun: ", psjun)

psmay = countOutliers(cc, "ps-may", 1)
print("Outliers in ps-may: ", psmay)

psapr = countOutliers(cc, "ps-apr", 1)
print("Outliers in ps-apr: ", psapr)

basep = countOutliers(cc, "ba-sep", 1)
print("Outliers in ba-sep: ", basep)

baaug = countOutliers(cc, "ba-aug", 1)
print("Outliers in ba-aug: ", baaug)

bajul = countOutliers(cc, "ba-jul", 1)
print("Outliers in ba-jul: ", bajul)

bajun = countOutliers(cc, "ba-jun", 1)
print("Outliers in ba-jun: ", bajun)

bamay = countOutliers(cc, "ba-may", 1)
print("Outliers in ba-may: ", bamay)

baapr = countOutliers(cc, "ba-apr", 1)
print("Outliers in ba-apr: ", baapr)

pasep = countOutliers(cc, "pa-sep", 1)
print("Outliers in pa-sep: ", pasep)

paaug = countOutliers(cc, "pa-aug", 1)
print("Outliers in pa-aug: ", paaug)

pajul = countOutliers(cc, "pa-jul", 1)
print("Outliers in pa-jul: ", pajul)

pajun = countOutliers(cc, "pa-jun", 1)
print("Outliers in pa-jun: ", pajun)

pamay = countOutliers(cc, "pa-may", 1)
print("Outliers in pa-may: ", pamay)

paapr = countOutliers(cc, "pa-apr", 1)
print("Outliers in pa-apr: ", paapr)'''
