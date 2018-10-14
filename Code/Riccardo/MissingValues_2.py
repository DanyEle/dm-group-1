import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def qqplot(data, y_label, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.probplot(data, dist="norm", plot=plt)
    ax.get_lines()[0].set_marker('.')
    plt.ylabel(y_label)
    plt.savefig(file_name, format = 'svg')
    plt.clf()
    

def my_crosstab(x, y, tit, leg):
    ct = pd.crosstab(x, y)
    pct = ct.div(ct.sum(1).astype(float), axis=0)
    pct.plot(kind='bar', stacked=True, title=tit)
    plt.legend(leg, loc='best')
    f_name = '_'.join(tit.split()) + '.svg'
    plt.savefig(f_name, format = 'svg')
    plt.clf()
    

def my_densplot(attr, tit, x_label):
    attr_values = sorted(cc[attr].unique())
    for v in attr_values:
        cc.age[cc[attr] == v].plot(kind='kde')
    plt.title(tit)
    plt.xlabel(x_label)
    plt.legend(attr_values, loc='best')
    f_name = '_'.join(tit.split()) + '.svg'
    plt.savefig(f_name, format = 'svg')
    plt.clf()



#load dataset into a dataframe
cc = pd.read_csv("C:\\Users\\Riccardo Manetti\\Desktop\\prova\\credit_default_train.csv")
cc_org = cc.copy()


#INITIAL PLOT
qqplot(cc.age, 'Age', 'age_initial.svg')


#SEX
#fill missing values with the mode of sex (female)
cc['sex'] = cc['sex'].fillna(cc['sex'].mode()[0])

sexes = sorted(cc['sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
cc['sex_val'] = cc['sex'].map(genders_mapping).astype(int)


#EDUCATION
cc['education'] = cc['education'].fillna('others')


#AGE
#dataframe without rows where age is -1
cc_age = cc[cc['age'] != -1]
cc['age'] = cc['age'].groupby([cc['sex'], cc['education']]).apply(lambda x: x.replace(-1, x.median()))


#STATUS
cc['status'] = cc['status'].groupby([cc['sex'], cc['education']]).apply(lambda x: x.fillna(x.mode()[0]))


#FINAL PLOT
#age_plot2
qqplot(cc.age, 'Age', 'age_final.svg')

#Age density plot by education
my_densplot('education', 'Age Density Plot by Education', 'Age')

#Age density plot by status
my_densplot('education', 'Age Density Plot by Status', 'Age')

#Credit default Rate by Gender
my_crosstab(cc['credit_default'], cc['sex_val'], \
            'Credit default Rate by Gender', sexes)

#Education Rate by Gender
my_crosstab(cc['education'], cc['sex_val'], \
            'Education default Rate by Gender', sexes)

#Status Rate by Gender
my_crosstab(cc['status'], cc['sex_val'], \
            'Status Rate by Gender', sexes)