import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


#imports for k-means
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




def main():
    #load dataset into a dataframe
    credit_cards = pd.read_csv("D:\dm-group-1\Dataset\credit_default_train.csv")

   # credit_cards = pd.read_csv("/home/daniele/dm-group-1/Dataset/credit_default_train.csv")
    #remember: load the corresponding function from Riccardo's scripts
    credit_cards = remove_missing_values(credit_cards)
        
    #firstly, create a data frame where we have three extra columns: ba, pa, ps
    #such attributes are the average values of the corresponding 6 attributes in the original data frame
    credit_cards_avg = create_data_frame_avg(credit_cards, ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"], ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"],  ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"])
    
    #we now need to convert categorical attributes into numerical attributes
    
    #convert education into numerical, as it is an ordinal attribute
    credit_cards_edu_numerical = convert_education_to_numerical_attribute(credit_cards_avg)
        
    attributes_k_means = ['limit', 'education', 'age', 'ba', 'ps', 'pa-apr', 'pa-may', 'pa-jun', 'pa-jul', 'pa-aug', 'pa-sep']
  
    #let's actually run k-means, shall we?  
    compute_k_means_given_data_frame(credit_cards_edu_numerical, 50, attributes_k_means)
        
    
    
def compute_k_means_given_data_frame(df, max_k, attributes):
    credit_cards_k_means = df[attributes]
    
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
        






