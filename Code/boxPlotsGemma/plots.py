#EXAMPLE OF USAGE: see file example.py

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
"""This function takes as an input the path of the file that represents the dataset (filePath) and the string which is the name of a column (colName). This function returns the column as a list or -1, if thecolumn doesn't have numerical values"""


def highlightColumn(filePath, colName):
    if ((colName == "sex") | (colName == "education") | (colName == "status") |
        (colName == "credit_default")):
        print("Cannot print boxplot from non numerical data\n")
        return -1
    if ((colName != "limit") & (colName != "age") & (colName != "ps-sep") &
        (colName != "ps-sep") & (colName != "ps-aug") & (colName != "ps-jul") &
        (colName != "ps-jun") & (colName != "ps-may") & (colName != "ps-apr") &
        (colName != "ba-sep") & (colName != "ba-sep") & (colName != "ba-aug") &
        (colName != "ba-jul") & (colName != "ba-jun") & (colName != "ba-may") &
        (colName != "ba-apr") & (colName != "pa-sep") & (colName != "pa-sep") &
        (colName != "pa-aug") & (colName != "pa-jul") & (colName != "pa-jun") &
        (colName != "pa-may") & (colName != "pa-apr")):
        print("Typo in the column name\n")
        return -1
    else:
        data = open(filePath, "r")
        if (colName == "limit"):
            col = 0
        if (colName == "age"):
            col = 4
        if (colName == "ps-sep"):
            col = 5
        if (colName == "ps-aug"):
            col = 6
        if (colName == "ps-jul"):
            col = 7
        if (colName == "ps-jun"):
            col = 8
        if (colName == "ps-may"):
            col = 9
        if (colName == "ps-apr"):
            col = 10
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


"""This function creates and saves a boxplot as "boxName.png", which has howMany different plots (whose names are in the string list called colNames) from the file whose path is filePath"""


def printPlots(boxName, filePath, howMany, colNames):
    if howMany == 0:
        return
    dataToPlot = []
    plt.clf()
    xNames = []
    for i in range(0, howMany):
        colName = colNames[i]
        currCol = highlightColumn(filePath, colName)
        dataToPlot.append(currCol)
        xNames.append(i + 1)
    fig = plt.figure(1, figsize=(9, 6))
    pix = fig.add_subplot(111)
    plt.boxplot(dataToPlot)
    plt.xticks(xNames, colNames)
    fig.savefig(boxName + '.svg', bbox_inches='tight')
    return
