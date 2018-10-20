#EXAMPLE OF USAGE: see file example.py
import math
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
"""This function takes as an input the path of the file that represents the dataset (filePath) and the string which is the name of a column (colName). This function returns the column as a list or -1, if thecolumn doesn't have numerical values"""


def highlightColumn(filePath, colName, pandas=None):
    if ((colName == "sex") | (colName == "education") | (colName == "status") |
        (colName == "credit_default") | (colName == "ps-sep") |
        (colName == "ps-aug") | (colName == "ps-jul") | (colName == "ps-jun") |
        (colName == "ps-may") | (colName == "ps-apr")):
        print("Cannot print boxplot from non numerical data\n")
        return -1
    if ((colName != "limit") & (colName != "age") & (colName != "ps-sep") &
        (colName != "ba-sep") & (colName != "ba-sep") & (colName != "ba-aug") &
        (colName != "ba-jul") & (colName != "ba-jun") & (colName != "ba-may") &
        (colName != "ba-apr") & (colName != "pa-sep") & (colName != "pa-sep") &
        (colName != "pa-aug") & (colName != "pa-jul") & (colName != "pa-jun") &
        (colName != "pa-may") & (colName != "pa-apr")):
        print("Typo in the column name\n")
        return -1
    else:
        if (pandas):
            data = filePath
            return getattr(data, colName)
        else:
            data = open(filePath, "r")
            if (colName == "limit"):
                col = 0
            if (colName == "age"):
                col = 4
            if (colName == "ba-sep"):
                col = 11
            if (colName == "ba-aug"):
                col = 12
            if (colName == "ba-jul"):
                col = 13
            if (colName == "ba-jun"):
                col = 14
            if (colName == "ba-may"):
                col = 15
            if (colName == "ba-apr"):
                col = 16
            if (colName == "pa-sep"):
                col = 17
            if (colName == "pa-aug"):
                col = 18
            if (colName == "pa-jul"):
                col = 19
            if (colName == "pa-jun"):
                col = 20
            if (colName == "pa-may"):
                col = 21
            if (colName == "pa-apr"):
                col = 22
            lines = [list(l.split(',')) for l in data]
            selectedCol = []
            #print("Colonna ", colName)
            for i in range(1, len(lines)):
                #print("colonna vera: ", lines[i][col])
                selectedCol.append(int(lines[i][col]))
                #print(selectedCol[i - 1])
    return selectedCol


"""This function creates and saves a boxplot as "boxName.figExtension", which has howMany different plots (whose names are in the string list called colNames) from the file whose path is filePath"""


def printPlots(boxName,
               figExtension,
               filePath,
               howMany,
               colNames,
               flag=None,
               pandas=None,
               log=None):
    if howMany == 0:
        return
    dataToPlot = []
    plt.clf()
    xNames = []
    for i in range(0, howMany):
        colName = colNames[i]
        currCol = highlightColumn(filePath, colName, pandas)
        if (log):
            if (not ((colName == "ba-sep") & (colName == "ba-aug") &
                     (colName == "ba-jul") & (colName == "ba-jun") &
                     (colName == "ba-may") & (colName == "ba-apr") &
                     (colName == "pa-sep") & (colName == "pa-aug") &
                     (colName == "pa-jul") & (colName == "pa-jun") &
                     (colName == "pa-may") & (colName == "pa-apr"))):
                #Since the logarithm is defined only fo positive numbers
                newCol = [math.log(int(x)) for x in currCol]
                dataToPlot.append(newCol)
                boxName = boxName + "Log"
        else:
            if flag:
                if colName == "limit":
                    myInt = 49989
                    newCol = [x / myInt for x in currCol]
                    dataToPlot.append(newCol)
                    boxName = boxName + "Salary"
            else:
                dataToPlot.append(currCol)
        xNames.append(i + 1)
    fig = plt.figure(1, figsize=(9, 6))
    pix = fig.add_subplot(111)
    boxPlot = plt.boxplot(dataToPlot)
    plt.xticks(xNames, colNames)
    pix.yaxis.grid(
        True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    fig.savefig(boxName + '.' + figExtension, bbox_inches='tight')
    return
