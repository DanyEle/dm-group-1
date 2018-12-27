import os
import sys
import math
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import collections

from scipy.stats.stats import pearsonr
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, '../../Daniele')
from k_means import convert_education_to_numerical_attribute
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def plotEpsMinPts(dataFrame, distance):
    path = distance + "/Curves/"
    if not os.path.exists(path):
        os.makedirs(path)
    columns = dataFrame.drop(["sex", "education", "status", "credit_default"],
                             axis=1)
    scaler = MinMaxScaler()  #normalization
    df1 = scaler.fit_transform(columns.values)  #df1 is normalized
    dist = squareform(pdist(df1, metric=distance))  #distance matrix
    mean_values = list()
    values_for_k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    #all curves in separate pdfs
    for k in values_for_k:
        kth_distances = []
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])
        filePath = path + "curvek" + str(k) + ".pdf"
        plt.clf()
        xLab = "Points sorted according to distance from " + str(
            k) + "th nearest neighbour"
        yLab = str(k) + "th nearest neighbour distance"
        plt.xlabel(xLab)
        plt.ylabel(yLab)
        plt.yticks(np.arange(0, 2, 0.1))
        ax = plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
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
    plt.savefig(path + "allcurves.pdf")
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
    columnsTitles = [
        "limit", "age", "education", "ps-sep", "ps-high", "ps-low", "ps-mode",
        "ba", "pa-sep", "pa-aug", "pa-jul", "pa-jun", "pa-may", "pa-apr"
    ]
    dataNew = dataNew[columnsTitles]
    return dataNew


"""
This function takes as an input dataframe and writes to file the correlation matrix of the columns considered important (obtained from the function dataFrameManipulation)
"""


def computeCorrelation(df, correlationPath):
    df = dataFrameManipulation(df)
    numCols = df.shape[1]
    with open(correlationPath, "w+") as currFile:
        print(" ".join(str(val) for val in list(df)), file=currFile)
        for i in range(0, numCols):
            row = []
            for j in range(0, numCols):
                x = df.iloc[:, i].tolist()
                y = df.iloc[:, j].tolist()
                row.append(float('%.1f' % (pearsonr(x, y)[0] * 100)))
            print(" ".join(str(val) for val in row), file=currFile)


"""This function runs DB scan on all the attributes of the data frame it receives as an input.
"""


def dbscan(dataFrame, eps, minpts, distance, version):
    if ((version == 0) or (version == 1)):
        dataFrame = dataFrameManipulation(dataFrame)
        if (
                version == 1
        ):  #Attributes: limit, age, education, ps-sep, ps-mode, ba_m, all pas
            dataFrame = dataFrame.drop(["ps-high", "ps-low"], axis=1)
    if (version == 2):  #Attributes: limit,ba_m, pa_m, ps_m
        toDrop = [
            "sex", "education", "status", "age", "ps-sep", "ps-aug", "ps-jul",
            "ps-jun", "ps-may", "ps-apr", "ba-sep", "ba-aug", "ba-jul",
            "ba-jun", "ba-may", "ba-apr", "pa-sep", "pa-aug", "pa-jul",
            "pa-jun", "pa-may", "pa-apr", "credit_default", "ps_mode", "Lab"
        ]
        dataFrame = dataFrame.drop(toDrop, axis=1)
    if (version == 3):  #Attributes: limit, all bas, all pas
        toDrop = [
            "sex", "education", "status", "age", "ps-sep", "ps-aug", "ps-jul",
            "ps-jun", "ps-may", "ps-apr", "credit_default", "ps_mode", "ba_m",
            "pa_m", "Lab", "ps_m"
        ]
        dataFrame = dataFrame.drop(toDrop, axis=1)
    if (version == 4):  #Attributes: limit, all bas
        toDrop = [
            "sex", "education", "status", "age", "ps-sep", "ps-aug", "ps-jul",
            "ps-jun", "ps-may", "ps-apr", "pa-sep", "pa-aug", "pa-jul",
            "pa-jun", "pa-may", "pa-apr", "credit_default", "ps_mode", "ba_m",
            "pa_m", "Lab", "ps_m"
        ]
        dataFrame = dataFrame.drop(toDrop, axis=1)
    scaler = MinMaxScaler()  #normalization
    print("Epsilon: " + str(eps) + " minPts: " + str(minpts))
    tenta = 0
    df1 = scaler.fit_transform(dataFrame.values)
    dbscan = DBSCAN(eps=eps, min_samples=minpts, metric=distance, p=1)
    dbscan.fit(df1)
    numClusters = len(set(dbscan.labels_))
    if (numClusters == 1):
        print("Only noise")
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
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(myPath)
    return


"""This function elaborates the outputs of runs of dbscan over different values of k and corresponding epsilon
"""


def EpsMinPtsEvaluation(eps,
                        minpts,
                        rootPath,
                        df2,
                        version=None,
                        distance=None,
                        noAge=None):
    d = ({})  #dictionary to store clusterings
    for i in range(0, len(eps)):
        if (noAge):
            df2 = df2.drop(["age"], axis=1)
        myDBscan, s, labels = dbscan(df2, eps[i], minpts[i], distance, version)
        d[eps[i]] = ({'Sil': s, 'Labels': labels})
        if (s == -2):
            continue
        dataNew = convert_education_to_numerical_attribute(df2)
        dataNew = newPSColumns(dataNew)
        nameCols = dataNew.columns
        #plot all possible cuts
        for attributeX in nameCols:
            for attributeY in nameCols:
                semiRootPath = rootPath + "eps" + str(eps[i]) + "k" + str(
                    minpts[i]) + "/"
                if not os.path.exists(semiRootPath):
                    os.makedirs(semiRootPath)
                plotClusters(
                    myDBscan, df2, attributeX, attributeY,
                    semiRootPath + attributeX + "_" + attributeY + ".pdf")
        print("All possible cuts plotted")
    pickle.dump(d, open(rootPath + "dictionary.p", "wb"))
    print("Completed serialization of dictionary")
    return


"""This function writes the information contained in the dictionary d into the opened file myFile"""


def printDictionary(d, myFile):
    print("eps NumClusters numNoise Silhouette", file=myFile)
    for k, v in d.items():
        labels = v["Labels"]
        numClusters = len(set(labels)) - (1 if -1 in labels else 0)
        numNoise = list(labels).count(-1)
        sil = v["Sil"]
        print(k, numClusters, numNoise, sil, file=myFile)


"""This function writes the dump of files path1 and path2 into a file results.txt in the same folder"""


def clusteringComp(metric, path1, path2):
    with open(path1, 'rb') as f:
        d = pickle.load(f)
        with open(metric + "/results.txt", "w+") as myFile:
            printDictionary(d, myFile)
    with open(path2, 'rb') as f:
        d = pickle.load(f)
        with open(metric + "/results.txt", "a") as myFile:
            printDictionary(d, myFile)


def bestClusters(n, path):
    with open(path, 'r') as f:
        d = ({})
        lines = [line.rstrip('\n') for line in f]
        for i in range(1, len(lines)):  #Excluding title line
            values = lines[i].split()
            sil = values[5]
            d[sil] = ({
                'Distance': values[0],
                'Version': values[1],
                'Eps': values[2],
                'numClusters': values[3],
                'numNoisePts': values[4]
            })
        #od = collections.OrderedDict(sorted(d.items()))
        print("The best ", n, "clusters are:")
        howMany = 0
        for i, key in enumerate(sorted(d.keys(), reverse=True)):
            if howMany > n - 1: break
            values = d[key]
            #Only clusterings that make more than 1 cluster are ok
            if (int(values['numClusters']) != 1):
                howMany = howMany + 1
                print("~~~~~~~~~~~~~~", howMany, "~~~~~~~~~~~~~~~~~")
                print("Distance: ", values['Distance'])
                print("Version: ", values['Version'])
                print("Eps: ", values['Eps'])
                print("numTrueClusters: ", values['numClusters'])
                print("numNoisePts: ", values['numNoisePts'])
                print("Silhouette: ", key)
