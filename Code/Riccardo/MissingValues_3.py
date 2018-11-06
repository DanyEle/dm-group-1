#import math
#import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


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


def qqplot(data, y_label, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.probplot(data, dist="norm", plot=plt)
    ax.get_lines()[0].set_marker('.')
    plt.ylabel(y_label)
    plt.savefig(file_name, format = 'pdf')
    plt.clf()


def my_crosstab(x, y, tit, leg):
    ct = pd.crosstab(x, y)
    pct = ct.div(ct.sum(1).astype(float), axis=0)
    pct.plot(kind='bar', stacked=True, title=tit)
    plt.legend(leg, loc='best')
    f_name = '_'.join(tit.split()) + '.pdf'
    plt.savefig(f_name, format = 'pdf')
    plt.clf()


def my_densplot(attr, tit, x_label):
    attr_values = sorted(cc_NoMV[attr].unique())
    for v in attr_values:
        cc_NoMV.age[cc_NoMV[attr] == v].plot(kind='kde')
    plt.title(tit)
    plt.xlabel(x_label)
    plt.legend(attr_values, loc='best')
    f_name = '_'.join(tit.split()) + '.pdf'
    plt.savefig(f_name, format = 'pdf')
    plt.clf()


def main():
    global cc_NoMV
    
    #load dataset into a dataframe
    cc = pd.read_csv('C:\\Users\\Richard\\Desktop\\DM_proj\\credit_default_train.csv')
    
    #INITIAL PLOT
    qqplot(cc.age, 'Age', 'age_initial.pdf')
    
    #REMOVE MISSING VALUES
    cc_NoMV = remove_missing_values(cc) 
    
    sexes = sorted(cc_NoMV['sex'].unique())
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
    cc_NoMV['sex_val'] = cc_NoMV['sex'].map(genders_mapping).astype(int)
    
    cc_values = sorted(cc_NoMV['credit_default'].unique())
    
    #FINAL PLOT
    #age_plot2
    qqplot(cc_NoMV.age, 'Age', 'age_final.pdf')
    
    #Age density plot by education
    my_densplot('education', 'Age Density Plot by Education', 'Age')
    
    #Age density plot by status
    my_densplot('education', 'Age Density Plot by Status', 'Age')
    
    #Gender Rate by Credit default 
    my_crosstab(cc_NoMV['sex'], cc_NoMV['credit_default'], 
                'Credit default Rate by Gender', cc_values)
    
    #Education Rate by Credit default 
    my_crosstab(cc_NoMV['education'], cc_NoMV['credit_default'], 
                'Education default Rate by Gender', cc_values)
    
    #Status Rate by Credit default
    my_crosstab(cc_NoMV['status'], cc_NoMV['credit_default'], 
                'Status Rate by Gender', cc_values)