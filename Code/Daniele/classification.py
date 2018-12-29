
from __future__ import absolute_import, division, print_function  # Python 2/3 compatibility

import warnings
#warnings.filterwarnings("ignore")
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample


from sklearn import svm



import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

## Import Keras objects for Deep Learning

from keras.models  import Sequential, K
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop

#stuff for decision trees
from sklearn.tree import DecisionTreeClassifier
import pydotplus 
from sklearn import tree
from IPython.display import Image


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier



def run_deep_classification_algs():
    
    """INITIALIZE INPUT DATA"""
    
    os.chdir('/home/daniele/dm-group-1/Code/Daniele')


    #import Gemma's function for removing outliers
    sys.path.insert(0, './../Gemma/Part 1/')
    from outliers import removeOutliers
    
    #import Riccardo's function for removing missing values
    sys.path.insert(0, './../Riccardo')
    from MissingValues_3 import remove_missing_values
    
    #import Daniele's function for converting education into a numerical attribute
    #import also Daniele's function for adding mean columns' value to the data frame
    from dependencies import create_data_frame_avg
    
    sys.path.insert(0, './../Maddalena')
    from formula_1_2_correction import correct_ps_values
    
    
    #initialize data frame with the attributes we wanna consider
    attributes = ["limit", "education", "sex", "status", "age",
                                'ps-apr', 'ps-may', 'ps-jun', 'ps-jul', 'ps-aug', 'ps-sep',
                                "ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep", 
                                "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"]
    
    #try to create a model based on the whole dataset
    url_train = "../../Dataset/credit_default_train.csv"
    #url_train = "../../Dataset/UCI_Credit_Card.csv"
    
    #training data frame
    credit_cards_deep_learning_train, labels_train = load_pre_process_dataset(url_train, True, attributes)

    #test dataframe, no labels in this case
    url_test = "../../Dataset/credit_default_test.csv"
    credit_cards_deep_learning_test = load_pre_process_dataset(url_test, False, attributes)
    
    #input training data
    X = credit_cards_deep_learning_train.values
    #input training labels
    y = labels_train
    
    #with no classifier, we could just say no customer has defaulted, and we would get 77.8% accuracy
    np.mean(y), np.mean(1-y)
        
    #split the dataset- 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11111)
    
    """DEEP LEARNING"""
    
    #We now need to normalize the data
    
    normalizer = StandardScaler()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)
        
    
    """DL Model 1 - One hidden layer"""
    #First model try: 1 single hidden layer with 14 hidden nodes. 
    model_1 = Sequential ([ Dense(20, input_shape=(23,), activation="relu"), Dense(1, activation="sigmoid")])
    
    model_1.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])
    
    #Acc = 0.8238. Val_Acc = 0.8290
    run_hist_1 = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=500)
    plot_model_train_validation_loss(run_hist_1)
    
    model_compute_test_validation_accuracy_DL(model_1, X_train, y_train)
    model_compute_test_validation_accuracy_DL(model_1, X_test, y_test)
    
    #output_model_results_to_file(model_1, "group_1_submission_26_DL_1_Node.txt", credit_cards_deep_learning_test, None)
    
    
    """DL Model 2 - two hidden layers"""
    
    #probably overfitting
    model_2 = Sequential([
        #3 hidden layers with 14 neurons in each one
        Dense(20, input_shape=(23,), activation="relu"),
        Dense(10, activation="relu"),
        #final layer
        Dense(1, activation="sigmoid"),
    ])  
    
    #0.723 on the test dataset :(
    model_2.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])
        
    run_hist_2 = model_2.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=200)
    plot_model_train_validation_loss(run_hist_2)
    
    #we should test on non-normalized data because the validation dataset on Kaggle is also unnormalized
    
    #validation on the training dataset
    model_compute_test_validation_accuracy_DL(model_2, X_train, y_train)
    
    #validation on the test dataset
    model_compute_test_validation_accuracy_DL(model_2, X_test, y_test)
    
    #output_model_results_to_file(model_2, "group_1_submission_24_DL_M2.txt", credit_cards_deep_learning_test, None)   


    """DECISION TREES - SDT"""
    dec_tree = DecisionTreeClassifier(criterion='gini', max_depth=2, 
                             min_samples_split=2, min_samples_leaf=1)
    
    dec_tree.fit(X_train, y_train)

    #let's visualize feature importance
    for col, imp in zip(attributes, dec_tree.feature_importances_):
        print(col, imp)   
    #we need to convert the decision tree's classes into labels again    
    model_compute_test_validation_accuracy(dec_tree, X_test, y_test)
    model_compute_test_validation_accuracy(dec_tree, X_train, y_train)

    dec_tree.classes_ = ["no", "yes"]
    
    dot_data = tree.export_graphviz(dec_tree, out_file=None,  
                                feature_names=attributes, 
                                class_names=dec_tree.classes_,  
                                filled=True, rounded=True,  
                                special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    Image(graph.create_png())
        
      #Stratifled shuffle split with decision trees
    """ SDT with 10-fold cross-validation"""
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
        
    i = 1
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        dec_tree = DecisionTreeClassifier(criterion='gini', max_depth=2, 
                             min_samples_split=2, min_samples_leaf=1)
        
        dec_tree.fit(X_train, y_train) 
        
        print("i = " + str(i))
        model_compute_test_validation_accuracy(dec_tree, X_test, y_test)
        model_compute_test_validation_accuracy(dec_tree, X_train, y_train)

        dec_tree.classes_ = ["no", "yes"]

        i = i + 1
        
    
    #plot the fourth element, starting from 0
    generator = sss.split(X, y)
    
    train_index, test_index = list(generator)[3]
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dec_tree = DecisionTreeClassifier(criterion='gini', max_depth=2, 
                         min_samples_split=2, min_samples_leaf=1)
    
    dec_tree.fit(X_train, y_train) 
    
    model_compute_test_validation_accuracy(dec_tree, X_test, y_test)
    model_compute_test_validation_accuracy(dec_tree, X_train, y_train)

    dec_tree.classes_ = ["no", "yes"]
    
    dot_data = tree.export_graphviz(dec_tree, out_file=None,  
                                feature_names=attributes, 
                                class_names=dec_tree.classes_,  
                                filled=True, rounded=True,  
                                special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    Image(graph.create_png())
    
    for col, imp in zip(attributes, dec_tree.feature_importances_):
        print(col, imp)   
    
    
    """ SDT with downsampling """
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    
    
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.2, random_state=11111)


    dec_tree = DecisionTreeClassifier(criterion='gini', max_depth=2, 
                             min_samples_split=2, min_samples_leaf=1)
    
    dec_tree.fit(X_train_res, y_train_res)
    
    model_compute_test_validation_accuracy(dec_tree, X_test_res, y_test_res)
    model_compute_test_validation_accuracy(dec_tree, X_train_res, y_train_res)
    
    """SDT with oversampling"""
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    
    dec_tree = DecisionTreeClassifier(criterion='gini', max_depth=2, 
                             min_samples_split=2, min_samples_leaf=1)
    
    dec_tree.fit(X_res, y_res)
    
    model_compute_test_validation_accuracy(dec_tree, X_test, y_test)
    model_compute_test_validation_accuracy(dec_tree, X_train, y_train)   
    
    
    """OPTIMIZED DECISION TREE with the whole dataset """
    
    ##Grid Search
    param_list_grid = {'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 100],
                      'min_samples_leaf': [1, 5, 10, 20, 30, 40, 50, 100],
                      'max_depth': [None] + list(np.arange(2, 5)), #previously, not considered this. 
                     }
    for i in range(1, 6):
        print("i = " + str(i))
        dec_tree_optimized_grid = optimize_model(dec_tree, 1, param_list_grid, X_train, y_train)
        model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_test, y_test)
        model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_train, y_train)
        
    ##Randomized Search
    param_list_rand_search = {'max_depth': [None] + list(np.arange(2, 100)),
              'min_samples_split': [2, 5, 10, 20, 30, 50, 100, 150, 200],
              'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100, 150, 200],
             }
    
    for i in range(1, 6):
        print("i = " + str(i))
        dec_tree_optimized_grid = optimize_model(dec_tree, 2, param_list_grid, X_train, y_train)
        model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_test, y_test)
        model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_train, y_train)
        
    
    """OPTIMIZED DECISION TREE with 10-fold cross-validation and grid search"""    
    
    #Grid Search
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
    
    param_list_grid = {'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 100],
                      'min_samples_leaf': [1, 5, 10, 20, 30, 40, 50, 100],
                      'max_depth': [None] + list(np.arange(2, 5)), #previously, not considered this. 
                     }
        
    i = 0
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
      
    
        print("i = " + str(i))
        dec_tree_optimized_grid = optimize_model(dec_tree, 1, param_list_grid, X_train, y_train)
        model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_test, y_test)
        model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_train, y_train)
        
        i = i + 1
        
    """OPTIMIZED DECISION TREE with 10-fold cross-validation and randomized search"""    
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
    param_list_rand_search = {'max_depth': [None] + list(np.arange(2, 100)),
              'min_samples_split': [2, 5, 10, 20, 30, 50, 100, 150, 200],
              'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100, 150, 200],
             }
        
    i = 1
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("i = " + str(i))
        dec_tree_optimized_rand_search = optimize_model(dec_tree, 2, param_list_rand_search, X_train, y_train)
        model_compute_test_validation_accuracy(dec_tree_optimized_rand_search, X_test, y_test)
        model_compute_test_validation_accuracy(dec_tree_optimized_rand_search, X_train, y_train)
        
        i = i + 1
        
    """ Output best ODT with grid /randomized search"""
        
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
    param_list_rand_search = {'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 100],
                      'min_samples_leaf': [1, 5, 10, 20, 30, 40, 50, 100],
                      'max_depth': [None] + list(np.arange(2, 100)), #previously, not considered this. 
                     }
    generator = sss.split(X, y)
    train_index, test_index = list(generator)[8]
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dec_tree_optimized_rand_search = optimize_model(dec_tree, 2, param_list_rand_search, X_train, y_train)
    model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_test, y_test)
    model_compute_test_validation_accuracy(dec_tree_optimized_grid, X_train, y_train)
    
    output_model_results_to_file(dec_tree_optimized_rand_search, "group_1_submission_29_dec_rand_search_59_levels.txt", credit_cards_deep_learning_test, None)

    
    #plot the best decision tree
    dec_tree_optimized_rand_search.classes_ = ["no", "yes"]
    dot_data = tree.export_graphviz(dec_tree_optimized_rand_search, out_file=None,  
                                feature_names=attributes, 
                                class_names=dec_tree.classes_,  
                                filled=True, rounded=True,  
                                special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    Image(graph.create_png())
    
    for col, imp in zip(attributes, dec_tree.feature_importances_):
        print(col, imp)   
        
        

            
    """SIMPLE RANDOM FOREST by 10-fold cross-validation"""
    
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
        
    i = 1
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print("i = " + str(i))
        rf_model = RandomForestClassifier(n_estimators=200, 
                                 criterion='gini', 
                                 max_depth=None, 
                                 min_samples_split=2, 
                                 min_samples_leaf=1, 
                                 class_weight=None)
    
        rf_model.fit(X_train, y_train)
        
        model_compute_test_validation_accuracy(rf_model, X_test, y_test)
        model_compute_test_validation_accuracy(rf_model, X_train, y_train)
        i = i + 1
        
    
    
    """OPTIMIZED RANDOM FOREST by 10-fold cross-validation with grid search"""
    rf_model = RandomForestClassifier(n_estimators=200, 
                                 criterion='gini', 
                                 max_depth=None, 
                                 min_samples_split=2, 
                                 min_samples_leaf=1, 
                                 class_weight=None)
    
    rf_model.fit(X_train, y_train)
        
    
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
        
    i = 1
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print("i = " + str(i))
    
        #let's try tuning the hyperparameters by grid search
        param_list_grid = {
              'min_samples_split': [2, 5, 10, 20, 30, 50, 100, 150, 200],
              'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100, 150, 200],
              'max_depth': [None] + list(np.arange(2, 5)),
                 }
        rf_model_optimized_grid = optimize_model(rf_model, 1, param_list_grid, X_train, y_train)
    
        model_compute_test_validation_accuracy(rf_model_optimized_grid, X_test, y_test)
        model_compute_test_validation_accuracy(rf_model_optimized_grid, X_train, y_train)
        
        i = i + 1

    
    #output_model_results_to_file(rf_model_optimized_grid, "group_1_submission_18_RF_grid.txt", credit_cards_deep_learning_test, None)
        
    
    """OPTIMIZED RANDOM FOREST by 10-fold cross-validation with randomized search"""
    
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
        
    i = 1
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print("i = " + str(i))
  
        param_list_rand_search = {'max_depth': [None] + list(np.arange(2, 100)),
              'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
              'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
             }

        #F1-score= 0.91; accuracy = 0.917; roc-auc = 0.982
        clf_optimized_rand_search = optimize_model(rf_model, 2, param_list_rand_search, X_train, y_train)
            
        model_compute_test_validation_accuracy(clf_optimized_rand_search, X_test, y_test)
        model_compute_test_validation_accuracy(clf_optimized_rand_search, X_train, y_train)

        i = i + 1
        
        
    """Optimized random forest with downsampling """
    
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    
    
    rf_model = RandomForestClassifier(n_estimators=200, 
                                 criterion='gini', 
                                 max_depth=None, 
                                 min_samples_split=2, 
                                 min_samples_leaf=1, 
                                 class_weight=None)
    
    rf_model.fit(X_train, y_train)
        
    """Get the the Optimized Random Forest Model with best performance for grid search"""
    
    
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
    param_list_rand_search = {'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 100],
                      'min_samples_leaf': [1, 5, 10, 20, 30, 40, 50, 100],
                      'max_depth': [None] + list(np.arange(2, 5)), #previously, not considered this. 
                     }
    generator = sss.split(X, y)
    train_index, test_index = list(generator)[3]
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf_optimized_rand_search = optimize_model(rf_model, 1, param_list_rand_search, X_train, y_train)
    model_compute_test_validation_accuracy(rf_optimized_rand_search, X_test, y_test)
    model_compute_test_validation_accuracy(rf_optimized_rand_search, X_train, y_train)
    
    
    output_model_results_to_file(rf_optimized_rand_search, "group_1_submission_30_optimized_rand_forest_grid.txt", credit_cards_deep_learning_test, None)
    
    
    
    """Get the the Optimized Random Forest Model with best performance for rand search """

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1111)
    param_list_rand_search = {'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 100],
                      'min_samples_leaf': [1, 5, 10, 20, 30, 40, 50, 100],
                      'max_depth': [None] + list(np.arange(2, 100)), #previously, not considered this. 
                     }
    generator = sss.split(X, y)
    train_index, test_index = list(generator)[3]
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf_optimized_rand_search = optimize_model(rf_model, 2, param_list_rand_search, X_train, y_train)
    model_compute_test_validation_accuracy(rf_optimized_rand_search, X_test, y_test)
    model_compute_test_validation_accuracy(rf_optimized_rand_search, X_train, y_train)
    
    output_model_results_to_file(rf_optimized_rand_search, "group_1_submission_31_optimized_rand_forest_rand_search.txt", credit_cards_deep_learning_test, None)

            
    
    """ K - NEAREST NEIGHBORS """
    """ KNN by Cross-validaiton """ #pretty poor performance

    indexes_vector = []
    accuracy_vector = []
    f1_vector = []
    
    for i in range(1, 100):
        clf = KNeighborsClassifier(n_neighbors=i)
        indexes_vector.append(i)
        scores = cross_val_score(clf, X, y, cv=10)
        print('KNN k = ' + str(i) + 'Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
        accuracy_vector.append(scores.mean())
    
        scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
        print('KNN k = ' + str(i) + 'F1-score: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
        f1_vector.append(scores.mean())

    
    """Naive Bayes"""
    
    model = GaussianNB()

    # Train the model using the training sets 
    model.fit(X_test, y_test)
    
    model_compute_test_validation_accuracy(model, X_test, y_test)

    
    
def inspect_training_set_for_na_infinite_values(X_train_sss):
    
    i = 0
    for row in X_train_sss:
        if not all(np.isfinite(row)):
            print("Row " + i + " is infinite")
            
        if any(np.isnan(row)):
            print("Row " + i + " does not have any NAs")
        i = i + 1
        
        if np.isinf(row).any():
            print("Row " + i + " is infinite")
            
                
    
def load_all_labels():
    #some random experiments
    
    
    url_train = "../../Dataset/UCI_Credit_Card.csv"
    
    attributes = ["limit", "age", "education",
                                'ps-apr', 'ps-may', 'ps-jun', 'ps-jul', 'ps-aug', 'ps-sep',
                                "ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep", 
                                "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep", "credit_default"]
    #training data frame
    credit_cards_deep_learning_train = load_pre_process_dataset(url_train, False, attributes)
        
    all_y_labels = credit_cards_deep_learning_train["credit_default"]
    
    
    credit_cards_deep_learning_train = credit_cards_deep_learning_train.drop(columns=["credit_default"])
    
    y_pred_class = clf_optimized_rand_search.predict(credit_cards_deep_learning_train.values) #HARD
    
    
    print('Accuracy on test dataset is {:.3f}'.format(accuracy_score(all_y_labels,y_pred_class)))

    
    
    
"""ALL THE FUNCTIONS REQUIRED FOR CLASSIFICATION UNDERNEATH"""

#optimization_method == 1 --> Grid
#optimization_method == 2 --> Randomized Search
def optimize_model(model, optimization_method, param_list, X, y):
    if(optimization_method == 1):
        grid_search = GridSearchCV(model, param_grid=param_list)
        grid_search.fit(X, y)
        clf = grid_search.best_estimator_
        report(grid_search.cv_results_, n_top=3)
        return(clf)
        
        
    elif(optimization_method == 2):
        random_search = RandomizedSearchCV(model, param_distributions=param_list, n_iter=100)
        random_search.fit(X, y)
        clf = random_search.best_estimator_
        report(random_search.cv_results_, n_top=3)
        return(clf)
        
     


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



def output_model_results_to_file(keras_model, file_name, credit_cards_deep_learning_test, normalizer):
    
    #firstly, normalize the test data set
    if(normalizer != None):
        X_data = normalizer.fit_transform(credit_cards_deep_learning_test)
    else:
        X_data = credit_cards_deep_learning_test
    
    y_labels = keras_model.predict(X_data)
    
    new_file = open(file_name, 'w')
    #just iterate over the two lists element by element and print them out
    new_file.write("index,credit_default" + "\n")
    for i in range(0, len(y_labels)):
        new_file.write(str(i) + "," + str(number_to_default(y_labels.item(i))) + "\n")
        
    #done, now close the file and stop outputting to it
    new_file.close()
        


    


def convert_credit_default_to_numerical_attribute(credit_cards_input):    
    credit_default_column = credit_cards_input["credit_default"]
    credit_default_column_new = []
   
    for default_row  in credit_default_column:
        #if found a string, convert string to number
        if not isinstance(default_row, int):
            credit_default_column_new.append(default_to_number(default_row))
        #if found a number, just leave the number be
        else:
            credit_default_column_new.append(default_row)
       
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
        #if a string, convert to integer
        if not isinstance(education_row, int):
            education_column_new.append(educ_category_to_number(education_row))
        else:
        #if already an integer, let it be
            education_column_new.append(education_row)
            
       
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



def convert_sex_status_to_numerical(df):
     #convert sex and status to numerical
    label_encoders = dict()
    column2encode = ['sex', 'status']
    
    for col in column2encode:    

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return(df)


#if train_dataset == True --> also remove missing values and outliers
#if train_dataset == False --> Do not remove missing values and outliers
def load_pre_process_dataset(url, train_dataset, attributes_deep_learning):
    #Load the training data
    credit_cards_df = pd.read_csv(url)

    #firstly, remove missing values
    credit_cards_no_missing_outliers = remove_missing_values(credit_cards_df)
    
    credit_cards_no_missing_outliers = convert_sex_status_to_numerical(credit_cards_no_missing_outliers)
    
    #credit_cards_no_missing_outliers = correct_ps_values(credit_cards_df)
    
    #and remove outliers (this function operates in place)
    if(train_dataset == True):
        removeOutliers(credit_cards_no_missing_outliers)
        #pass
        
    #BUG FIXED: also need to re-adjust the indexes after removing outliers
    credit_cards_no_missing_outliers.index = np.arange(0, len(credit_cards_no_missing_outliers))

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



def plot_model_train_validation_loss(keras_model):
       #plot accuracy of second deep learning model
    fig, ax = plt.subplots()
    ax.plot(keras_model.history["loss"],'r', marker='.', label="Train Loss")
    ax.plot(keras_model.history["val_loss"],'b', marker='.', label="Validation Loss")
    ax.legend()
    
    
    
def model_compute_test_validation_accuracy(keras_model, X_test, y_test):

    y_pred_class = keras_model.predict(X_test) #HARD
    y_pred_prob = keras_model.predict_proba(X_test) #SOFT    
    
    y_pred_class = convert_float_array_to_int_array(y_pred_class)

    # Print model performance and plot the roc curve
    print('Accuracy: ' + str(round(accuracy_score(y_test,y_pred_class), 2)) + " Roc-auc " + str(round(roc_auc_score(y_test,y_pred_prob[:,1]), 2)))
    #plot_roc(y_test, y_pred_prob, 'NN')
    print(classification_report(y_test, y_pred_class))
    

def model_compute_test_validation_accuracy_DL(keras_model, X_test, y_test):

    y_pred_class = keras_model.predict(X_test) #HARD    
    y_pred_prob = keras_model.predict_proba(X_test) #HARD    

    y_pred_class = convert_float_array_to_int_array(y_pred_class)
    y_pred_prob = convert_float_array_to_int_array(y_pred_prob)

    # Print model performance and plot the roc curve
    print('Accuracy on test dataset is {:.3f}'.format(accuracy_score(y_test,y_pred_class)))
    
    print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob)))

    print(classification_report(y_test, y_pred_class))
    

    
    
    
def convert_float_array_to_int_array(y_pred_class):
    y_pred_output = []
    for class_row in y_pred_class:
        #higher probability to be in class 0
        if(class_row <= 0):
            y_pred_output.append(0)
        else:
            y_pred_output.append(1)
            
    return(y_pred_output)
    
    
    


def number_to_default(number):
    if number == 0:
        return "no"
    elif number == 1:
        return "yes"