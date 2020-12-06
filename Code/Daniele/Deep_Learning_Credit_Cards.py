
# coding: utf-8


#Preliminaries

from __future__ import absolute_import, division, print_function  # Python 2/3 compatibility

import warnings
warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


import seaborn as sns

# In[2]:


## Import Keras objects for Deep Learning

from keras.models  import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop


# In[163]:


#import Gemma's function for removing outliers
"""Questa funzione elimina gli outlier individuati tramite una visual analysis in place nel data frame dataFrame"""


def removeOutliers(dataFrame):
    print("Initial size of data frame: ", dataFrame.shape)
    baMay = getattr(dataFrame, "ba-may")
    baApr = getattr(dataFrame, "ba-apr")
    paAug = getattr(dataFrame, "pa-aug")
    paApr = getattr(dataFrame, "pa-apr")
    paMay = getattr(dataFrame, "pa-may")
    rows = []
    for i in range(0, len(baMay)):
        if ((int(baMay[i]) < -5000) | (int(baApr[i]) < -5000) |
            (int(paAug[i]) > 500000) | (int(paApr[i]) > 500000) |
            (int(paMay[i]) > 400000)):
            rows.append(i)
    print("Visual analysis, number of rows to be dropped: ", len(rows))
    dataFrame.drop(dataFrame.index[rows], inplace=True)
    print("Final size of data frame: ", dataFrame.shape)
    return

#import Riccardo's function for removing missing values
def remove_missing_values(df_in):
    df_out = df_in
    #SEX
    #fill missing values with the mode of sex (female)
    df_out['sex'] = df_out['sex'].fillna(df_out['sex'].mode()[0])
    
    #EDUCATION
    df_out['education'] = df_out['education'].fillna('others')
    
    #STATUS
    df_out['status'] = df_out['status'].groupby([df_out['sex'], df_out['education']]).apply( 
      lambda x: x.fillna(x.mode()[0]))
    
    #AGE
    #dataframe without rows where age is -1
    df_out['age'] = df_out['age'].groupby([df_out['sex'], df_out['education'],
      df_out['status']]).apply(lambda x: x.replace(-1, x.median()))
    
    return df_out

#import Daniele's function for converting education into a numerical attribute
#import also Daniele's function for adding mean columns' value to the data frame
def create_data_frame_avg(credit_cards, columns_balance, columns_pa, columns_ps):
    mean_ba_column = credit_cards[columns_balance].mean(axis=1)
    mean_pa_column = credit_cards[columns_pa].mean(axis=1)
    mean_ps_column =  credit_cards[columns_ps].mean(axis=1)
    
    credit_cards["ps"] = mean_ps_column
    credit_cards["pa"] = mean_pa_column
    credit_cards["ba"] = mean_ba_column
    
    return(credit_cards)

# In[164]:


def convert_credit_default_to_numerical_attribute(credit_cards_input):    
    credit_default_column = credit_cards_input["credit_default"]
    credit_default_column_new = []
   
    for default_row  in credit_default_column:
        credit_default_column_new.append(default_to_number(default_row))
       
    credit_cards_input["credit_default"] = credit_default_column_new
    return credit_cards_input

def default_to_number(category):
    if category == "no":
        return 0
    elif category == "yes":
        return 1 
    
    
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


# In[165]:


#if train_dataset == True --> also remove missing values and outliers
#if train_dataset == False --> Do not remove missing values and outliers
def load_pre_process_dataset(url, train_dataset, attributes_deep_learning):
    #Load the training data
    credit_cards_df = pd.read_csv(url)

    #firstly, remove missing values
    credit_cards_no_missing_outliers = remove_missing_values(credit_cards_df)
    #and remove outliers (this function operates in place)
    if(train_dataset == True):
        removeOutliers(credit_cards_no_missing_outliers)
    #create mean value columns
    credit_cards_avg = create_data_frame_avg(credit_cards_no_missing_outliers, ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"], ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"],  ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"])
    credit_cards_edu_numerical = convert_education_to_numerical_attribute(credit_cards_avg)
    #and convert the credit_default into a numerical attribute as well
    if(train_dataset == True):
        credit_cards_default_num = convert_credit_default_to_numerical_attribute(credit_cards_edu_numerical)
    else:
        credit_cards_default_num = credit_cards_edu_numerical
    #pick the attributes you wanna use for deep learning
    credit_cards_deep_learning = credit_cards_default_num[attributes_deep_learning]
    
    if(train_dataset == True):    
        return(credit_cards_deep_learning, credit_cards_edu_numerical["credit_default"])
    else:
        return(credit_cards_deep_learning)
#lovely python lets me return multiple values <3


# In[185]:


attributes_deep_learning = ["limit", "age", "education", "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep", 'ps-apr', 'ps-may', 'ps-jun', 'ps-jul', 'ps-aug', 'ps-sep', "ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep", "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"]
url_train = "../../Dataset/credit_default_train.csv"

credit_cards_deep_learning_train, labels_train = load_pre_process_dataset(url_train, True, attributes_deep_learning)

df = credit_cards_deep_learning_train

df["label"] = labels_train

#let's notice that the dataset is actually strongly imbalanced
df["label"].value_counts()

#let's downsample the majority class
# Separate majority and minority classes
df_majority = df[df["label"]==0]
df_minority = df[df["label"]==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=2208,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_downsampled["label"].value_counts()


credit_cards_deep_learning_train = df_downsampled




# Take a peek at the data 
print(credit_cards_deep_learning_train.shape)
credit_cards_deep_learning_train.sample(5)


# In[186]:


url_test = "../../Dataset/credit_default_test.csv"
credit_cards_deep_learning_test = load_pre_process_dataset(url_test, False, attributes_deep_learning)

print(credit_cards_deep_learning_test.shape)
credit_cards_deep_learning_test.sample(5)


# In[187]:


#Input: X is the dataframe without the credit_default label


y = credit_cards_deep_learning_train["label"]

credit_cards_deep_learning_train.drop(columns=["label"], inplace=True)



X = credit_cards_deep_learning_train.values
#the output consists of the state with diabetes


# In[188]:


np.mean(y), np.mean(1-y)


# In[189]:


#now let's split the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)


# Above, we see that about 22.1% of customers have defaulted, while 77.8% do not.  This means we can get an accuracy of 77.8% without any model - just declare that no one has defaulted. We will calculate the ROC-AUC score to evaluate performance of our model, and also look at the accuracy as well to see if we improved upon the 77.8%% accuracy.
# ## Exercise: Get a baseline performance using Random Forest
# To begin, and get a baseline for classifier performance:
# 1. Train a Random Forest model with 200 trees on the training data.
# 2. Calculate the accuracy and roc_auc_score of the predictions.

# In[190]:


## Train the RF Model
rf_model = RandomForestClassifier(n_estimators=200)
#we train both with X input data and Y input data
rf_model.fit(X_train, y_train)


# In[191]:


# Make predictions on the test set - both "hard" predictions, and the scores (percent of trees voting yes)
y_pred_class_rf = rf_model.predict(X_test) #HARD
y_pred_prob_rf = rf_model.predict_proba(X_test) #SOFT


print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_rf)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_rf[:,1])))


# In[192]:


def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])


plot_roc(y_test, y_pred_prob_rf[:, 1], 'RF')


# ## Build a Single Hidden Layer Neural Network
# 
# We will use the Sequential model to quickly build a neural network.  Our first network will be a single layer network.  We have 8 variables, so we set the input shape to 8.  Let's start by having a single hidden layer with 12 nodes.

# In[248]:


def plot_model_train_validation_loss(keras_model):
       #plot accuracy of second deep learning model
    fig, ax = plt.subplots()
    ax.plot(keras_model.history["loss"],'r', marker='.', label="Train Loss")
    ax.plot(keras_model.history["val_loss"],'b', marker='.', label="Validation Loss")
    ax.legend()
    
def model_compute_test_validation_accuracy(keras_model):    
    y_pred_class_nn_1 = keras_model.predict_classes(X_test_norm)
    y_pred_prob_nn_1 = keras_model.predict(X_test_norm)

    # Print model performance and plot the roc curve
    print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1)))
    print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1)))
    plot_roc(y_test, y_pred_prob_nn_1, 'NN')
    
def output_labels_given_model_data(keras_model, credit_cards_deep_learning_test):
    #firstly, normalize the test set
    X_data = normalizer.fit_transform(credit_cards_deep_learning_test)
    
    y_labels = keras_model.predict_classes(X_data)
    
    list_elements = np.arange(0, len(y_labels))

    
    #and now let's co
    
    return(list_elements, y_labels)  


# In[193]:


## First let's normalize the data
## This aids the training of neural nets by providing numerical stability
## Random Forest does not need this as it finds a split only, as opposed to performing matrix multiplications


normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)


# In[197]:


# Define the Model 
# Input size is 9-dimensional
# 1 hidden layer, 12 hidden nodes, sigmoid activation
# Final layer has just one node with a sigmoid activation (standard for binary classification)


model_1 = Sequential([
    Dense(12, input_shape=(27,), activation="relu"),
    Dense(1, activation="sigmoid")
])


# In[198]:


#  This is a nice tool to view the model you have created and count the parameters

 #probably overfitting
model_2 = Sequential([
    #3 hidden layers with 14 neurons in each one
    Dense(20, input_shape=(27,), activation="relu"),
    Dense(10, activation="relu"),
    #final layer
    Dense(1, activation="sigmoid"),
])  

#0.723 on the test dataset :(
model_2.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])
    
run_hist_2 = model_2.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=200)
plot_model_train_validation_loss(run_hist_2)
# In[203]:


## Like we did for the Random Forest, we generate two kinds of predictions
#  One is a hard decision, the other is a probabilitistic score.

plot_model_train_validation_loss(run_hist_2)

plot_roc(y_test, y_pred_prob_nn_1, 'NN')


# Let's look at the `run_hist_1` object that was created, specifically its `history` attribute.

# In[147]:


run_hist_1.history.keys()


# Let's plot the training loss and the validation loss over the different epochs and see how it looks.

# In[204]:


fig, ax = plt.subplots()
ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()


# Looks like the losses are still going down on both the training set and the validation set.  This suggests that the model might benefit from further training.  Let's train the model a little more and see what happens. Note that it will pick up from where it left off. Train for 1000 more epochs.

# In[205]:


## Note that when we call "fit" again, it picks up where it left off
run_hist_1b = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1000)


# In[206]:


n = len(run_hist_1.history["loss"])
m = len(run_hist_1b.history['loss'])
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(range(n), run_hist_1.history["loss"],'r', marker='.', label="Train Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["loss"], 'hotpink', marker='.', label="Train Loss - Run 2")

ax.plot(range(n), run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["val_loss"], 'LightSkyBlue', marker='.',  label="Validation Loss - Run 2")

ax.legend()


# In[210]:


model_compute_test_validation_accuracy(model_1)


# In[253]:


list_labels, y_labels = output_labels_given_model_data(model_1, credit_cards_deep_learning_test)
#len(y_labels)



def number_to_default(number):
    if number == 0:
        return "no"
    elif number == 1:
        return "yes"

def output_results_to_file(file_name, y_labels):
    sys.stdout = open(file_name, 'w')
    #just iterate over the two lists element by element
    print("index,credit_default")
    for i in range(0, len(y_labels)):
        print(str(i) + "," + str(number_to_default(y_labels.item(i))))



output_results_to_file("group_1_submission_3rd_try.txt", y_labels)


# Note that this graph begins where the other left off.  While the training loss is still going down, it looks like the validation loss has stabilized (or even gotten worse!).  This suggests that our network will not benefit from further training.  What is the appropriate number of epochs?

# ## Exercise
# Now it's your turn.  Do the following in the cells below:
# - Build a model with two hidden layers, each with 6 nodes
# - Use the "relu" activation function for the hidden layers, and "sigmoid" for the final layer
# - Use a learning rate of .003 and train for 1500 epochs
# - Graph the trajectory of the loss functions, accuracy on both train and test set
# - Plot the roc curve for the predictions
# 
# Experiment with different learning rates, numbers of epochs, and network structures

# In[258]:

# Define the Model 
# Input size is 8-dimensional
# 1 hidden layer, 6 hidden nodes, relu activation
# 1 hidden layer, 6 hidden nodes, relu activation
# Final layer has just one node with a sigmoid activation (standard for binary classification)


model_2 = Sequential([
    #hidden layers
    Dense(6, input_shape=(27,), activation="relu"),
    Dense(6, activation="relu"),
    #final layer
    Dense(1, activation="sigmoid")
])


# In[259]:


model_2.summary()


# In[260]:


model_2.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])


# In[263]:


#Train function!!
run_hist_2 = model_2.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1500)

# the fit function returns the run history. 
# It is very convenient, as it contains information about the model fit, iterations etc.


# In[270]:


#model_compute_test_validation_accuracy(model_2)

#file
#sys.stdout = open(file_name, 'w')


y_pred_class_nn_1 = model_2.predict_classes(X_test_norm)
y_pred_prob_nn_1 = model_2.predict(X_test_norm)

# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1)))


# In[274]:



y_pred_class_nn_1 = model_2.predict_classes(X_test_norm)
y_pred_prob_nn_1 = model_2.predict(X_test_norm)

# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1)))


list_labels, y_labels = output_labels_given_model_data(model_2, credit_cards_deep_learning_test)
#len(y_labels)



def number_to_default(number):
    if number == 0:
        return "no"
    elif number == 1:
        return "yes"

def output_results_to_file(file_name, y_labels):
    sys.stdout = open(file_name, 'w')
    #just iterate over the two lists element by element
    print("index,credit_default")
    for i in range(0, len(y_labels)):
        print(str(i) + "," + str(number_to_default(y_labels.item(i))))



output_results_to_file("group_1_submission_3rd_try.txt", y_labels)

