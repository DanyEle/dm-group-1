import os
import sys
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as p

from dbscan import plotEpsMinPts
from dbscan import EpsMinPtsEvaluation
from dbscan import computeCorrelation
from dbscan import clusteringComp
from dbscan import bestClusters


def clusteringV0(distance, path):
    rootPath = path + "ClustersV0/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, df, 0, distance)


def clusteringV1(distance, path):
    rootPath = path + "ClustersV1/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, df, 1, distance)


def clusteringNoAge(distance, path):
    rootPath = path + "ClustersNoAge/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, df, 1, distance, True)


def clusteringV2(distance, path):
    rootPath = path + "ClustersV2/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, dfMadda, 2, distance)


def clusteringV3(distance, path):
    rootPath = path + "ClustersV3/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, dfMadda, 3, distance)


def clusteringV4(distance, path):
    rootPath = path + "ClustersV4/"
    EpsMinPtsEvaluation(eps, minpts, rootPath, dfMadda, 4, distance)


filePath = "../../../Dataset/credit_default_cleaned.csv"
df = pd.read_csv(filePath)
filePathMadda = "../../../Dataset/dataMadda.csv"
dfMadda = pd.read_csv(filePathMadda)
#Compute correlation between attributes
#corrFile = "correlation.txt"
#computeCorrelation(df, corrFile)

distance = sys.argv[1]

#Plotting distance curves
#plotEpsMinPts(df, distance)

path = distance + "/"
if not os.path.exists(path + "ClustersV0/"):
    os.makedirs(path + "ClustersV0/")

if not os.path.exists(path + "ClustersV1/"):
    os.makedirs(path + "ClustersV1/")

if not os.path.exists(path + "ClustersNoAge/"):
    os.makedirs(path + "ClustersNoAge/")

if not os.path.exists(path + "ClustersV2/"):
    os.makedirs(path + "ClustersV2/")

if not os.path.exists(path + "ClustersV3/"):
    os.makedirs(path + "ClustersV3/")

if not os.path.exists(path + "ClustersV4/"):
    os.makedirs(path + "ClustersV4/")

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

#clusteringV0(distance, path)
#clusteringV1(distance, path)
#clusteringNoAge(distance, path)
#clusteringV2(distance, path)
clusteringV3(distance, path)
#clusteringV4(distance, path)

cPathV0 = distance + "/ClustersV0/dictionary.p"
cPathV1 = distance + "/ClustersV1/dictionary.p"
cPathNoAge = distance + "/ClustersNoAge/dictionary.p"
cPathV2 = distance + "/ClustersV2/dictionary.p"
cPathV3 = distance + "/ClustersV3/dictionary.p"
cPathV4 = distance + "/ClustersV4/dictionary.p"

#clusteringComp(distance, cPath, cPathV0)
#clusteringComp(distance, cPath, cPathV1)
#clusteringComp(distance, cPath, cPathNoAge)
#clusteringComp(distance, cPath, cPathV2)
#clusteringComp(distance, cPath, cPathV3)
#clusteringComp(distance, cPath, cPathV4)

outputPath = "allResults.txt"
#bestClusters(20, outputPath)
