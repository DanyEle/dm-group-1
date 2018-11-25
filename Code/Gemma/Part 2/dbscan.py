import os
import sys
import math
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, '../../Daniele')
from k_means import convert_education_to_numerical_attribute
from sklearn.cluster import DBSCAN


def paramTuning(dataFrame):
    columns = dataFrame.drop(["sex", "education", "status", "credit_default"],
                             axis=1)
    scaler = MinMaxScaler()  #normalization
    df1 = scaler.fit_transform(columns.values)  #df1 is normalized
    dist = squareform(pdist(df1, metric="euclidean"))  #distance matrix
    mean_values = list()
    values_for_k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4092]
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
        plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
        plt.savefig(filePath)
        print("Finito k = ", k)

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
        filePath = "Tuning/curvek" + str(k) + ".pdf"
        plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
        print("Finito k = ", k)
    plt.savefig("Tuning/allcurves.pdf")
    return


def dbscan(dataFrame, eps, minpts):
    dataFrame = dataFrame.drop(["sex", "status", "credit_default"], axis=1)
    dataNew = convert_education_to_numerical_attribute(dataFrame)
    scaler = MinMaxScaler()  #normalization
    df1 = scaler.fit_transform(dataNew.values)  #df1 is normalized
    dbscan = DBSCAN(eps=eps, min_samples=minpts)
    dbscan.fit(df1)
    if not os.path.exists("Clusters/"):
        os.makedirs("Clusters/")
    path = "Clusters/"
    numCols = df1.shape[1]
    for i in range(0, numCols):
        for j in range(i, numCols):
            c1 = dataNew.iloc[i]
            c2 = dataNew.iloc[j]
            plt.clf()
            plt.scatter(c1, c2)  #, c=dbscan.labels_, s=20)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.savefig(path + str(i) + str(j) + ".pdf")
    return
