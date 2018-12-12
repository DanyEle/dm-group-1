import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os




#imports for k-means
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from dependencies import *



def run_daniele_k_means_certain_attributes():
    
    os.chdir('/home/daniele/dm-group-1/Code/Daniele')

    #dependencies on Riccardo's remove missing values function
    sys.path.insert(0, './../Riccardo')
    from MissingValues_3 import remove_missing_values
    
    #dependencies on Gemma's remove outliers function
    sys.path.insert(0, './../Gemma/Part 1')
    from outliers import removeOutliers
    
    #dependencies on Maddalena's formula to correct ps values
    sys.path.insert(0, './../Maddalena')
    from formula_1_2_correction import correct_ps_values

    
    
    
    #load dataset into a dataframe
    credit_cards = pd.read_csv("./../../Dataset/credit_default_train.csv")

   # credit_cards = pd.read_csv("/home/daniele/dm-group-1/Dataset/credit_default_train.csv")
    #remember: load the corresponding function from Riccardo's scripts
    credit_cards = remove_missing_values(credit_cards)
    
    removeOutliers(credit_cards)
    
    #credit_cards = correct_ps_values(credit_cards)
    #firstly, create a data frame where we have three extra columns: ba, pa, ps
    #such attributes are the average values of the corresponding 6 attributes in the original data frame
    credit_cards_avg = create_data_frame_avg(credit_cards, ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"], ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"],  ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"])
    
    #convert education into numerical, as it is an ordinal attribute
    credit_cards_edu_numerical = convert_education_to_numerical_attribute(credit_cards_avg)
        
    ###ITERATION 1
    attributes_k_means_iter_1 = ['limit', 'education', 'age', 'ba', 'ps', 'pa-apr', 'pa-may', 'pa-jun', 'pa-jul', 'pa-aug', 'pa-sep']

    k_means_knee_method_means_given_data_frame(credit_cards_edu_numerical, 30, attributes_k_means_iter_1)
    k_means_knee_method_means_given_data_frame(credit_cards_edu_numerical, 10, attributes_k_means_iter_1)
    
    k_means_given_data_frame_k(credit_cards_edu_numerical, 8, attributes_k_means_iter_1, False)
    k_means_given_data_frame_k(credit_cards_edu_numerical, 4, attributes_k_means_iter_1, False)

    ###ITERATION 2

    attributes_k_means_iter_2 = ['limit', 'education', 'age', 'ps-apr', 'ps-may', 'ps-jun', 'ps-jul', 'ps-aug', 'ps-sep']
    
    k_means_knee_method_means_given_data_frame(credit_cards_edu_numerical, 30, attributes_k_means_iter_2)
    k_means_knee_method_means_given_data_frame(credit_cards_edu_numerical, 10, attributes_k_means_iter_2)
    
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 8, attributes_k_means_iter_2, False)
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 4, attributes_k_means_iter_2, False)

    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_k_means_iter_2, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)


    ###ITERATION 3:
    #let's test some attributes and compare them with the results obtained in iteration 2
    #2 attributes
    attributes_2 = ["ps-may", "ps-jun"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 9, attributes_2, False)  
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_2, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)


    
    #3 attributes
    attributes_3 = ["ps-jul", "ps-aug", "ps-sep"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 9, attributes_3, False)  
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_3, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)
    
    #4 attributes
    attributes_4 = ["ps-jul", "ps-aug", "ps-sep"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 9, attributes_4, False)  
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_4, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)
    
    #5 attributes
    attributes_5 = ["ps-apr","ps-may","ps-jun","ps-jul","ps-sep"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 9, attributes_5, False) 
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_5, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)
    
    #6 attributes    
    attributes_6 = ["ps-apr","ps-may","ps-jun","ps-jul","ps-sep"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 9, attributes_6, False) 
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_6, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)
    
    
    #7 attributes
    attributes_7 = ["age","ps-apr","ps-may","ps-jun","ps-jul","ps-aug","ps-sep"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 8, attributes_7, False)  
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_7, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)
    
    
    #8 attributes
    attributes_8 = ["age","limit", "ps-apr","ps-may","ps-jun","ps-jul","ps-aug","ps-sep"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 8, attributes_8, False)  
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_8, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)
    
    
    #9 attributes    
    attributes_9 = ["age","limit", "ps-apr","ps-may","ps-jun","ps-jul","ps-aug","ps-sep"]
    centers, kmeans, X, scaler = k_means_given_data_frame_k(credit_cards_edu_numerical, 8, attributes_9, False)  
    k_means_view_centroids(centers, kmeans, X, credit_cards_edu_numerical, attributes_9, scaler)    
    k_means_view_distribution(credit_cards_edu_numerical, kmeans)
    



    
    
    
    
def run_maddalena_k_means_experiment():
     #load dataset into a dataframe
    credit_cards = pd.read_csv("/home/daniele/dm-group-1/Dataset/credit_default_train.csv")
    credit_cards = remove_missing_values(credit_cards)
    removeOutliers(credit_cards)
    credit_cards_avg = create_data_frame_avg(credit_cards, ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"], ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"],  ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"])
    credit_cards_edu_numerical = convert_education_to_numerical_attribute(credit_cards_avg)
    credit_cards_k_means = credit_cards_edu_numerical
    credit_cards_k_means = credit_cards_k_means[['limit', 'education', 'age', 'ps-apr', 'ps-may', 'ps-jun', 'ps-jul', 'ps-aug', 'ps-sep']]
    
    #credit_cards_k_means = credit_cards_k_means.drop(columns=["sex", "credit_default", "status"])
    
    minK = 2
    maxK = 10
    
    sys.stdout = open("experiments_minK_" + str(minK) + "_maxK_" + str(maxK) + "_attributes_" + str(len(credit_cards_k_means.columns)) + ".txt", "w")
    for i in range(2, len(credit_cards_k_means.columns)):
        print("Amount of columns: " + str(i))
        results_i = kmeans_(credit_cards_k_means, minK, maxK, i)
        print_results(credit_cards_k_means, results_i, minK)
        
    #fine, now let's plot the 

    
    #result = pickle.load(open("kmeans_columns_3.p", "rb"))
    #kmeans_(credit_cards_k_means[["pa", "ba", "ps"]], 2, 4, 3)
    #print_results(credit_cards_k_means[["pa", "ba", "ps"]], result, 2)
    
        
    
    
def k_means_knee_method_means_given_data_frame(df, max_k, attributes):
    df = df[attributes]
    
    #first step: normalize the data applying a min-max scaling
    #such that it will be in the range 0-1
    scaler = MinMaxScaler()
    #X is the training data transformed
    X = scaler.fit_transform(df.values)
        #let's compute the SSE for all the clusters with k =2..50
    sse_list = list()
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100)
        kmeans.fit(X)
        
        sse = kmeans.inertia_
        sse_list.append(sse)
        
    
    #and now let's plot the results we got based on the knee method
    plt.plot(range(2, len(sse_list) + 2), sse_list)
    plt.ylabel('SSE', fontsize=22)
    plt.xlabel('K', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.show()
    plt.savefig("D:\dm-group-1\Code\Daniele\k_means_knee.pdf")
    
    
    
    
def k_means_given_data_frame_k(df, k, attributes, inverse_transform):
    df = df[attributes]
    #just run k-means once with the k passed. 
    scaler = MinMaxScaler()
    #X is the training data transformed
    X = scaler.fit_transform(df.values)
    
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100)
    kmeans.fit(X)
    
    #let's see the labels obtained
    np.unique(kmeans.labels_, return_counts=True)
    
    if(inverse_transform):
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        centers = kmeans.cluster_centers_
        
    return centers, kmeans, X, scaler
   
    
    
def k_means_view_centroids(centers, kmeans, X, credit_cards, attributes, scaler):
    
    credit_cards = credit_cards[attributes]
    #visualize clusters by parallel coordinates
    plt.figure(figsize=(8, 4))
    for i in range(0, len(centers)):
        plt.plot(centers[i], marker='o', label='Cluster %s' % i)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.xticks(range(0, len(credit_cards.columns)), credit_cards.columns, fontsize=18, rotation=90)
    plt.legend(fontsize=5)
    plt.show()
    #show_center_values_per_cluster_attributes(centers, attributes, credit_cards, scaler, kmeans)
    
    print('SSE %s' % kmeans.inertia_)
    print('Silhouette %s' % silhouette_score(X, kmeans.labels_))
    
    hist, bins = np.histogram(kmeans.labels_, 
                          bins=range(0, len(set(kmeans.labels_)) + 1))
    
     #amount of elements per cluster
    print("Amount of elements per cluster:")
    print(dict(zip(bins, hist)))
    
    
    
def k_means_view_distribution(credit_cards, kmeans):
    plot_crosstab_given_attribute(credit_cards, "credit_default",  kmeans.labels_)
    plot_crosstab_given_attribute(credit_cards, "sex", kmeans.labels_)
    plot_crosstab_given_attribute(credit_cards, "status", kmeans.labels_)
    
    #scatter plots
    plt.scatter(credit_cards['age'], credit_cards['ps'], c=kmeans.labels_, s=20)
    plt.scatter(credit_cards['limit'], credit_cards['ps'], c=kmeans.labels_, s=20)
    
    credit_cards["Label"] = kmeans.labels_
    
    plot_histogram(credit_cards, "age")
    plot_histogram(credit_cards, "ps")
    plot_histogram(credit_cards, "limit")
    plot_histogram(credit_cards, "education")
    



        
    
 
    
def plot_histogram(df, attribute):
    max_val = max(np.unique(df['Label']))
    for i in range(0, int(max_val)):
       plt.hist(df[df['Label']==i][attribute], alpha=0.5, label=str(i))

    plt.legend()
    plt.show()

    
    
def plot_crosstab_given_attribute(credit_cards, attribute, labels):
    
    pd.crosstab(credit_cards[attribute], labels)
    crosstab = pd.crosstab( labels, credit_cards[attribute])
    crosstab_normalized = crosstab.div(crosstab.sum(1).astype(float), axis=0)
    crosstab_normalized.plot(kind='bar', stacked=True, 
                   title='Clustering occurrences by ' + str(attribute) + ' class')
    
    pd.crosstab(credit_cards[attribute],  labels)
    
        
    
def show_center_values_per_cluster_attributes(centers, attributes, credit_cards, scaler, kmeans):
    
    credit_cards["Label"] = kmeans.labels_
    
    #loop through every single cluster
    centers = scaler.inverse_transform(centers)
    #i is the index of centers
    for i in range(0, len(centers)):
        print("Cluster " + str(i) + ":")
        print("Amount of elements in cluster " + str(i) + " is " + str(len(credit_cards[credit_cards["Label"] == i ]))  ) 
        amount_defaults_in_cluster = len(credit_cards.loc[(credit_cards["credit_default"] == "yes") & (credit_cards["Label"] == i)])
        amount_default_not_in_cluster = len(credit_cards.loc[(credit_cards["credit_default"] == "no") & (credit_cards["Label"] == i)])
        print("Amount of default elements in cluster " + str(i) + " is " + str(amount_defaults_in_cluster))
        print("Amount of non-default elements in cluster " + str(i) + " is " + str(amount_default_not_in_cluster))


        
        print("Ratio of default in cluster i (defaults/non-defaults) " + str((amount_defaults_in_cluster / amount_default_not_in_cluster)))
        print("-------------------------")
        #and loop through every single attribute of the cluster's center
        for j in range(0, len(centers[i])):
            print(attributes[j])
            print(round(centers[i][j], 2))
        print("-------------------------")
    

def convert_education_to_numerical_attribute(credit_cards_input):    
    education_column = credit_cards_input["education"]
    education_column_new = []
   
    for education_row  in education_column:
        education_column_new.append(educ_category_to_number(education_row))
       
    credit_cards_input["education"] = education_column_new
    return credit_cards_input

def educ_category_to_number(category):
    if category == "others":
        return 0
    elif category == "high school":
        return 1
    elif category == "university":
        return 2
    elif category == "graduate school":
        return 3
        
def correct_ps_values(data):
    #Correzione FORMULA 1
    #Sistemo valori -1
    for i in range(0, len(data)):
        for j in range(12, 17):
            ba=data.iat[i, j]
            if(ba>0):
                pa=data.iat[i, j+5]
                if((ba-pa)<=0):
                    ps=data.iat[i, j-6]
                    if(ps!=-1):
                        data.iloc[i,j-6]=-1
                        
        
    #Correzione FORMULA 2
    #sistemo valori -2
    for i in range(0, len(data)):
        for j in range(11,16):
            ba=data.iat[i,j]
            precBa=data.iat[i,j+1]
            pa=data.iat[i,j+6]
            ps=data.iat[i,j-6]
            if(ba<=0):
                if(ba==(precBa-pa)):
                    if(ps!=-2):
                        data.iloc[i,j-6]=-2
                        
                        
    return(data)


