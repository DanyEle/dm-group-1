import os
import sys
import math
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, '../../Daniele')
from k_means import convert_education_to_numerical_attribute
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def plotEpsMinPts(dataFrame):
    columns = dataFrame.drop(["sex", "education", "status", "credit_default"],
                             axis=1)
    scaler = MinMaxScaler()  #normalization
    df1 = scaler.fit_transform(columns.values)  #df1 is normalized
    dist = squareform(pdist(df1, metric="euclidean"))  #distance matrix
    mean_values = list()
    values_for_k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    #all curves in separate pdfs
    for k in values_for_k:
        kth_distances = []
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])
        filePath = "curvek" + str(k) + ".pdf"
        plt.clf()
        xLab = "Points sorted according to distance from " + str(
            k) + "th nearest neighbour"
        yLab = str(k) + "th nearest neighbour distance"
        plt.xlabel(xLab)
        plt.ylabel(yLab)
        plt.yticks(np.arange(0, 2, 0.1))
        ax = plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
        #ax.yaxis.tick_right()
        #ax.yaxis.set_label_position("right")
        plt.grid(color='black', linestyle='-', linewidth=1)
        plt.savefig(filePath)
        print("Plotted k = ", k)
    print("Starting plot of all curves")
    #all curves in one pdf
    for k in values_for_k:
        kth_distances = []
        xLab = "Points sorted according to distance from k-th nearest neighbour"
        yLab = "k-th nearest neighbour distance"
        plt.xlabel(xLab)
        plt.ylabel(yLab)
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])
        plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
        print("Iteration ", k)
    plt.savefig("allcurves.pdf")
    return


"""This function takes a data frame and adds to it the following variations of PS:
    (1) PS_low
    (2) PS_high
    (3) PS_mode
"""


def newPSColumns(df):
    psLow = []
    psHigh = []
    psMode = []
    for index, row in df.iterrows():
        psRow = [
            row['ps-sep'], row['ps-aug'], row['ps-jul'], row['ps-jun'],
            row['ps-may'], row['ps-apr']
        ]
        psLow.append(min(psRow))
        psHigh.append(max(psRow))
        psMode.append(max(set(psRow), key=psRow.count))
    df["ps-high"] = psHigh
    df["ps-low"] = psLow
    df["ps-mode"] = psMode
    return df


"""
This function takes a data frame and inserts the following columns:
    (1) BA average
    (2) Education converted to numerical value
    (3) PS smallest value
    (4) PS biggest value
    (5) PS mode
It drops:
    (a) BA of each month
    (b) PS of each month except for PS_sep
    (c) Education non numerical
    (d) status
    (e) credi_default
    (f) sex
At the end it returns the new data frame
"""


def dataFrameManipulation(df):
    df["ba"] = df[["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug",
                   "ba-sep"]].mean(axis=1)
    df = df.drop([
        "sex", "status", "credit_default", "ba-apr", "ba-may", "ba-jun",
        "ba-jul", "ba-aug", "ba-sep"
    ],
                 axis=1)
    dataNew = convert_education_to_numerical_attribute(df)
    dataNew = newPSColumns(dataNew)
    dataNew = dataNew.drop(["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug"],
                           axis=1)
    return dataNew


"""This function runs DB scan on all the attributes of the data frame it receives as an input.
"""


def dbscan(dataFrame, eps, minpts):
    dataFrame = dataFrameManipulation(dataFrame)
    scaler = MinMaxScaler()  #normalization
    print("Epsilon: " + str(eps))
    tenta = 0
    #df_att = dataFrame[c.split(",")]
    df1 = scaler.fit_transform(dataFrame.values)
    dbscan = DBSCAN(eps=eps, min_samples=minpts)
    dbscan.fit(df1)
    if (len(dbscan.labels_) == 1):
        print("Questa combinazione non funziona")
        s = -2
    else:
        s = silhouette_score(df1, dbscan.labels_)
    return dbscan, s, dbscan.labels_


"""
This function plots the scatterplot of the clustering on the couple of attributes attributeX and attributeY
"""


def plotClusters(dbscan, dataNew, attributeX, attributeY, myPath):
    plt.clf()
    plt.scatter(
        dataNew[attributeX], dataNew[attributeY], c=dbscan.labels_, s=10)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.tight_layout()
    plt.savefig(myPath)
    return


"""This function elaborates the outputs of runs of dbscan over different values of k and corresponding epsilon
"""


def EpsMinPtsEvaluation(eps, minpts, rootPath, df2):
    myFile = rootPath + "comparison.txt"
    with open(myFile, "w+") as currFile:
        for i in range(0, len(eps)):
            myDBscan, silhouette, labels = dbscan(df2, eps[i], minpts[i])
            nameCols = df2.columns
            for attributeX in nameCols:
                for attributeY in nameCols:
                    semiRootPath = rootPath + "cluster" + str(i) + "/"
                    if not os.path.exists(semiRootPath):
                        os.makedirs(semiRootPath)
                    plotClusters(
                        myDBscan, df2, attributeX, attributeY,
                        semiRootPath + attributeX + "_" + attributeY + ".pdf")
            print("Plotted cluster where minPts is ", minpts[i])
            print(
                "~~~~~~~~~~~Epsilon " + str(eps[i]) + "~~~~~~~~~~~",
                file=currFile)
            print("Silhouette: " + str(silhouette), file=currFile)
            print("Labels: ", labels, file=currFile)
    return
