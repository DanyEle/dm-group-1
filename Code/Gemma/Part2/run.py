import os
import sys
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as p

sys.path.insert(0, '../../Daniele')
from dbscan import plotEpsMinPts
from dbscan import EpsMinPtsEvaluation
from dbscan import computeCorrelation
from dbscan import clusteringComp
from dbscan import bestClusters


def clusteringv1(distance, path):
    rootPath = path + "Clusters/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, df, False, distance)


def clusteringv2(distance, path):
    rootPath = path + "ClustersShort/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, df, True, distance)


def clusteringNoAge(distance, path):
    rootPath = path + "ClustersNoAge/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, df, True, distance, True)


filePath = "../../../Dataset/credit_default_cleaned.csv"
df = pd.read_csv(filePath)

#Compute correlation between attributes
#corrFile = "correlation.txt"
#computeCorrelation(df, corrFile)

distance = sys.argv[1]

#Plotting distance curves
#plotEpsMinPts(df, distance)

path = distance + "/"
if not os.path.exists(path + "Clusters/"):
    os.makedirs(path + "Clusters/")

if not os.path.exists(path + "ClustersShort/"):
    os.makedirs(path + "ClustersShort/")

if not os.path.exists(path + "ClustersNoAge/"):
    os.makedirs(path + "ClustersNoAge/")

minpts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

if (distance == "euclidean"):
    eps = [
        0.23, 0.25, 0.28, 0.3, 0.32, 0.35, 0.38, 0.42, 0.46, 0.5, 0.55, 0.7,
        0.8
    ]
if (distance == "cityblock"):
    eps = [0.63, 0.65, 0.7, 0.8, 0.85, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.9, 2.4]
if (distance == "cosine"):
    eps = [
        0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.13, 0.15, 0.18, 0.2, 0.25,
        0.35
    ]
if (distance == "minkowski"):
    eps = [
        0.25, 0.27, 0.3, 0.33, 0.35, 0.39, 0.41, 0.45, 0.5, 0.6, 0.66, 0.75,
        0.8
    ]

#clusteringv1(distance, path)
#clusteringv2(distance, path)
clusteringNoAge(distance, path)

cPath = distance + "/Clusters/dictionary.p"
cPathShort = distance + "/ClustersShort/dictionary.p"
cPathNoAge = distance + "/ClustersNoAge/dictionary.p"

#clusteringComp(distance, cPath, cPathShort)

outputPath = "allResults.txt"
#bestClusters(20, outputPath)
