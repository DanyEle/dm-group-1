import os
from plots import printPlots
import pandas as pd

#Small dataset for head
#filePath = "../../Dataset/credit_default_train_small.csv"

filePath = "../../Dataset/credit_default_train.csv"
newpath = "boxplots/"
if not os.path.exists(newpath):
    os.makedirs(newpath)
figExtension = "png"
#figExtension = "pdf"
#figExtension = "svg"

#Example of boxplot of limit values
limitNames = ["limit"]
limitBox = newpath + "limit"
howManyLimit = 1
printPlots(limitBox, figExtension, filePath, howManyLimit, limitNames)
cc = pd.read_csv(filePath)
printPlots(limitBox, figExtension, cc, howManyLimit, limitNames, None, 1)

#Example of boxplot of limit values last flag activated
limitNames = ["limit"]
limitBox = newpath + "limitS"
howManyLimit = 1
printPlots(limitBox, figExtension, filePath, howManyLimit, limitNames, 1)
#Use flag 1 at the end (default none) to obtain the limit boxplot as
#function of the average income of Taiwan (NT$49,989, about US$1,700)

#Example of boxplot of age values
ageNames = ["age"]
ageBox = newpath + "age"
howManyAge = 1
printPlots(ageBox, figExtension, filePath, howManyAge, ageNames)

#Example of boxplot of ps values
psNames = ["ps-sep", "ps-aug", "ps-jul", "ps-jun", "ps-may", "ps-apr"]
psBox = newpath + "ps"
howManyPs = 6
printPlots(psBox, figExtension, filePath, howManyPs, psNames)

#Example of boxplot of ba values
baNames = ["ba-sep", "ba-aug", "ba-jul", "ba-jun", "ba-may", "ba-apr"]
baBox = newpath + "ba"
howManyBa = 6
printPlots(baBox, figExtension, filePath, howManyBa, baNames)

#Example of boxplot of pa values
paNames = ["pa-sep", "pa-aug", "pa-jul", "pa-jun", "pa-may", "pa-apr"]
paBox = newpath + "pa"
howManyPa = 6
printPlots(paBox, figExtension, filePath, howManyPa, paNames)
