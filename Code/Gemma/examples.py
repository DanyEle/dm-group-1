import os
import sys
from plots import printPlots
from plots import plotNumOutliers
sys.path.insert(0, '../Riccardo')
from MissingValues_2 import remove_missing_values
from outliers import countOutliers
from outliers import removeOutliers
import pandas as pd


def generatePlots(newpath, filePath, pandas=None, log=None):
    #Example of boxplot of limit values
    limitNames = ["limit"]
    limitBox = newpath + "limit"
    howManyLimit = 1
    printPlots(limitBox, figExtension, filePath, howManyLimit, limitNames,
               None, pandas, log)

    #Example of boxplot of limit values last flag activated
    printPlots(limitBox, figExtension, filePath, howManyLimit, limitNames, 1,
               pandas, log)
    #Use flag 1 at the end (default none) to obtain the limit boxplot as
    #function of the average income of Taiwan (NT$49,989, about US$1,700)

    #Example of boxplot of age values
    ageNames = ["age"]
    ageBox = newpath + "age"
    howManyAge = 1
    printPlots(ageBox, figExtension, filePath, howManyAge, ageNames, None,
               pandas, log)

    #Example of boxplot of ba values
    baNames = ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"]
    baBox = newpath + "ba"
    howManyBa = 6
    printPlots(baBox, figExtension, filePath, howManyBa, baNames, None, pandas,
               None)
    #logarith is undefined for negative values

    #Example of boxplot of ba values reverse order
    baNamesReverse = [
        "ba-sep", "ba-aug", "ba-jul", "ba-jun", "ba-may", "ba-apr"
    ]
    baBox = newpath + "baReverse"
    howManyBa = 6
    printPlots(baBox, figExtension, filePath, howManyBa, baNamesReverse, None,
               pandas, None)
    #logarith is undefined for negative values

    #Example of boxplot of pa values
    paNames = ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"]
    paBox = newpath + "pa"
    howManyPa = 6
    printPlots(paBox, figExtension, filePath, howManyPa, paNames, None, pandas,
               None)

    #Example of boxplot of pa values in reverse order
    paNamesReverse = [
        "pa-sep", "pa-aug", "pa-jul", "pa-jun", "pa-may", "pa-apr"
    ]
    paBox = newpath + "paReverse"
    howManyPa = 6
    printPlots(paBox, figExtension, filePath, howManyPa, paNamesReverse, None,
               pandas, None)
    return


def countAll(cc, k):
    numOutliers = []
    rows = []
    totalLength = len(cc.limit)
    print("Lines in dataset: ", totalLength)

    limit = countOutliers(cc, "limit", k)
    print("Outliers in limit: ", limit)
    numOutliers.append(limit)
    rows.append(10000)

    age = countOutliers(cc, "age", 1)
    print("Outliers in age: ", age)
    numOutliers.append(age)
    rows.append(10000)
    """pssep = countOutliers(cc, "ps-sep", k)
    print("Outliers in ps-sep: ", pssep)
    numOutliers.append(pssep)
    
    psaug = countOutliers(cc, "ps-aug", k)
    print("Outliers in ps-aug: ", psaug)
    numOutliers.append(psaug)
    
    psjul = countOutliers(cc, "ps-jul", k)
    print("Outliers in ps-jul: ", psjul)
    numOutliers.append(psjul)

    psjun = countOutliers(cc, "ps-jun", k)
    print("Outliers in ps-jun: ", psjun)
    numOutliers.append(psjun)
    
    psmay = countOutliers(cc, "ps-may", k)
    print("Outliers in ps-may: ", psmay)
    numOutliers.append(psmay)
    
    psapr = countOutliers(cc, "ps-apr", k)
    print("Outliers in ps-apr: ", psapr)
    numOutliers.append(psapr)"""

    basep = countOutliers(cc, "ba-sep", k)
    print("Outliers in ba-sep: ", basep)
    numOutliers.append(basep)
    rows.append(10000)

    baaug = countOutliers(cc, "ba-aug", k)
    print("Outliers in ba-aug: ", baaug)
    numOutliers.append(baaug)
    rows.append(10000)

    bajul = countOutliers(cc, "ba-jul", k)
    print("Outliers in ba-jul: ", bajul)
    numOutliers.append(bajul)
    rows.append(10000)

    bajun = countOutliers(cc, "ba-jun", k)
    print("Outliers in ba-jun: ", bajun)
    numOutliers.append(bajun)
    rows.append(10000)

    bamay = countOutliers(cc, "ba-may", k)
    print("Outliers in ba-may: ", bamay)
    numOutliers.append(bamay)
    rows.append(10000)

    baapr = countOutliers(cc, "ba-apr", k)
    print("Outliers in ba-apr: ", baapr)
    numOutliers.append(baapr)
    rows.append(10000)

    pasep = countOutliers(cc, "pa-sep", k)
    print("Outliers in pa-sep: ", pasep)
    numOutliers.append(pasep)
    rows.append(10000)

    paaug = countOutliers(cc, "pa-aug", k)
    print("Outliers in pa-aug: ", paaug)
    numOutliers.append(paaug)
    rows.append(10000)

    pajul = countOutliers(cc, "pa-jul", k)
    print("Outliers in pa-jul: ", pajul)
    numOutliers.append(pajul)
    rows.append(10000)

    pajun = countOutliers(cc, "pa-jun", k)
    print("Outliers in pa-jun: ", pajun)
    numOutliers.append(pajun)
    rows.append(10000)

    pamay = countOutliers(cc, "pa-may", k)
    print("Outliers in pa-may: ", pamay)
    numOutliers.append(pamay)
    rows.append(10000)

    paapr = countOutliers(cc, "pa-apr", k)
    print("Outliers in pa-apr: ", paapr)
    numOutliers.append(paapr)
    rows.append(10000)

    plotNumOutliers(numOutliers, rows, "pic.pdf")
    return


#Small dataset for head
#filePath = "../../Dataset/credit_default_train_small.csv"

filePath = "../../Dataset/credit_default_train.csv"
newPath = "boxplots/"
newPathNoMV = "boxplots/NoMissingValues/"
newPathNoO = "boxplots/NoOutliers/"
if not os.path.exists(newPathNoO):
    os.makedirs(newPathNoO)
if not os.path.exists(newPath):
    os.makedirs(newPath)
if not os.path.exists(newPathNoMV):
    os.makedirs(newPathNoMV)

#figExtension = "png"
figExtension = "pdf"
#figExtension = "svg"
dataFrame = pd.read_csv(filePath)
deep1 = dataFrame.copy()
deep2 = dataFrame.copy()
dataFrameNoMV = remove_missing_values(deep1)
dataNew = removeOutliers(deep2)

#Plotting dataset
generatePlots(newPath, filePath)

#Plotting dataset without missing values
generatePlots(newPathNoMV, dataFrameNoMV, 1)

print("Shape of dataset without missing values: ", dataFrameNoMV.shape)
#Plotting dataset without missing values on a logarithmic scale
generatePlots(newPathNoMV, dataFrameNoMV, 1, 1)

#Plotting dataset without missing values and outliers
print("Shape of dataset without outliers: ", dataNew.shape)
generatePlots(newPathNoO, dataNew, 1)

#Counting outliers
k = 1.5
countAll(dataFrameNoMV, k)
