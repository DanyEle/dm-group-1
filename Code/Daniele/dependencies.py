import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math




def main():
    #load dataset into a dataframe
    credit_cards = pd.read_csv("/home/daniele/dm-group-1/Dataset/credit_default_train.csv")
    
    column_names = credit_cards.columns
    columns_balance = ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"]
    columns_pa = ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"]
    columns_ps = ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"]
    
    #Show the April - September average balance of a person'
    plot_average_given_columns(columns_balance)
    compute_mean_std_for_columns(columns_balance, credit_cards)
    
    #Show the April - September average revenue of a person (fatturato?)
    plot_average_given_columns(columns_pa)
    compute_mean_std_for_columns(columns_pa, credit_cards)
    
    #Show the April - September average PS of a person (what's that?)
    plot_average_given_columns(columns_ps)
    compute_mean_std_for_columns(columns_ps, credit_cards)
    
    #Compute balance in the bank account based on the education level
    #Expected dependency: the higher the degree, the higher the balance 
    #Actual dependency: not completely true! People that have gone to graduate school may have lots of loans to pay back
    #so their balance is actually smaller than people who attended university
    compute_average_value_given_attribute_and_columns(credit_cards, columns_balance, "education")
    
    #The fact above is further confirmed by the fact that the average age of people in the dataset is 32 y.o.
    #so, they haven't had much time to "accumulate" money
    compute_mean_std_for_columns(["age"], credit_cards)
    
    #Compute salary of people based on their education level.
    #Expended dependency: the higher the degree, the higher the salary
    #Actual dependency: this is true. In fact, people attending graduate school specialize in a field and 
    #consequently earn a higher salary, on average. 
    compute_average_value_given_attribute_and_columns(credit_cards, columns_pa, "education")
    
    #Now check the gender bias: are men paid more than women?
    #this is not very much appreciable from the plot, however:
    #female = 5274.564074933687; male= 5331.044812133747
    #average age of female: 31, average age of men: 33
    compute_average_value_given_attribute_and_columns(credit_cards, columns_pa, "education")
    
    #now check relation between credit_default and salary
    compute_average_value_given_attribute_and_columns(credit_cards, columns_pa, "credit_default")
    #we can clearly see that people who had a default earn less
    
    
    compute_average_value_given_attribute_and_columns(credit_cards, columns_balance, "credit_default")
    
    #there is no such big difference when it comes to the the balance though


    
    #now check people earning the most money based on the age. 
    compute_average_value_given_attribute_and_columns(credit_cards, columns_pa, "age")
    
    #And check who has the most money in their bank account.
    compute_average_value_given_attribute_and_columns(credit_cards, columns_balance, "age")
    
   # Indeed, the older, the more money they have


    #now try to make a scatter plot between the salary and the balance
    scatter_plot_attribute_columns(columns_pa, columns_balance, data_frame, True)
    
    #we could also check the correlation between these attributes
    
    scatter_plot_attribute_columns(columns_pa, ["status"], data_frame, False)
    
    #and the correlation between these attributes too


    

    
    
    
def scatter_plot_attribute_columns(columns_x, columns_y, data_frame, log_notation):
    #compute the mean value of each columns' group
    
    if(log_notation == True):
        mean_values_x = np.log(data_frame[columns_x].mean(axis=1))
        mean_values_y = np.log(data_frame[columns_y].mean(axis=1))
    else:
        mean_values_x = data_frame[columns_x].mean(axis=1))
        mean_values_y = data_frame[columns_y]
        
        plt.scatter(x = mean_values_x, y = mean_values_y)
    

    

#Input: data_frame: A Data frame loaded with data loaded from a dataset
#       columns_df: A set of columns taken from the data frame, containing a set of months 

#example call: 
#compute_average_value_given_attribute_and_columns(credit_cards, columns_balance, "education")
def compute_average_value_given_attribute_and_columns(data_frame, columns_df, attribute):
    #firstly, group by education type
    group_attribute = data_frame.groupby(attribute)
    
    #get the mean value of all rows in each column[just columns, no rows]
    mean_attribute_value = group_attribute.mean()
    
    #now perform another group by according to the unique values of the attribute column passed
    #ex: group by the different education types if passing "education"
    grouped_values_by_attribute = mean_attribute_value[columns_df]
    
    #values_per_unique_column_entry contains the mean value of every type of unique attribure over allthe different months
    #considered 
    #ex: mean of the columns grouped by the education type 
    values_per_unique_column_entry = []
    
    for index in range(len(grouped_values_by_attribute)):
        mean_attribute_type = 0
        for column in columns_df:
            mean_attribute_type += grouped_values_by_attribute[index:(index+1)][column]
        mean_attribute_type = mean_attribute_type / len(columns_df)
        values_per_unique_column_entry.append(mean_attribute_type.mean())
       
    #now get the different unique types found
    unique_column_values = list(grouped_values_by_attribute.index)
    
   # for i in range(len(values_per_unique_column_entry)):
    #    print("Average value of " + unique_column_values[i] + " is " + str(values_per_unique_column_entry[i]))
        
     
    plt.bar(x=unique_column_values, height=values_per_unique_column_entry)
     
     
    

def plot_average_given_columns(list_columns):    
    
    mean_list = []
    for column in list_columns:
        mean_column = credit_cards[column].mean()
        mean_list.append(mean_column)
        
    months = ["April", "May", "June", "July", "August", "September"]
    plt.scatter(x=months, y = mean_list)
        
 
#Mean and standard deviation
def compute_mean_std_for_columns(list_columns, data_frame):
    mean_list = []
    std_list = []
    for column in list_columns:
        mean_column = data_frame[column].mean()
        std_column = data_frame[column].std()
        mean_list.append(mean_column)
        std_list.append(std_column)
    
    overall_mean = sum(mean_list) / len(mean_list)
    overall_std = sum(std_list) / len(std_list)
    
    print("Overall mean value of input columns is " + str(overall_mean))
    print("Mean STD value of input columns is " + str(overall_std))

    
    
    









