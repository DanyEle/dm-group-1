import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import Counter


#load dataset into a dataframe
cc = pd.read_csv("C:\\Users\\Riccardo Manetti\\Desktop\\prova\\credit_default_train.csv")

cc_col = cc.columns
columns_ba = ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"]
columns_pa = ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"]
columns_ps = ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"]

#counters
c_limit = Counter(cc.limit)
c_age = Counter(cc.age)
c_education = Counter(cc.education)
c_sex = Counter(cc.sex)
c_status = Counter(cc.status)

#AGE
#age_plot1
stats.probplot(cc.age, dist="norm", plot=plt)
plt.savefig('age_plt1.png')
plt.clf()

#calculate avg of age -1 values exclude
age_avg = cc[cc.age != -1].median().age

#replace age field -1 with the age_avg
cc.age = cc.age.replace(-1, age_avg)

#age_plot2
stats.probplot(cc.age, dist="norm", plot=plt)
plt.savefig('age_plt2.png')
