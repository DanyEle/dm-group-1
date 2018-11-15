import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#imports for k-means
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph




def main():
    #load dataset into a dataframe
    credit_cards = pd.read_csv("/home/daniele/dm-group-1/Dataset/credit_default_train.csv")
    #remember: load the corresponding function from Riccardo's scripts
    credit_cards = remove_missing_values(credit_cards)
        
    #firstly, create a data frame where we have three extra columns: ba, pa, ps
    #such attributes are the average values of the corresponding 6 attributes in the original data frame
    credit_cards_avg = create_data_frame_avg(credit_cards, ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"], ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"],  ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"])
    
    #we now need to convert categorical attributes into numerical attributes
    
    #convert education into numerical, as it is an ordinal attribute
    credit_cards_edu_numerical = convert_education_to_numerical_attribute(credit_cards_avg)
    
    attributes_k_means = ['limit', 'education', 'status', 'age', 'ps', 'pa']
    credit_cards_k_means = credit_cards_avg[attributes_k_means]
    
    
    
    
    

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
        






