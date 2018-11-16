import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

#DATA CORRECTION
def findMinusOne(data):
    count=0
    rows=set()
    tot=0
    for i in range(0, len(data)):
        for j in range(12, 17):
            ba=data.iat[i, j]
            if(ba>0):
                pa=data.iat[i, j+5]
                if((ba-pa)<=0):
                    ps=data.iat[i, j-6]
                    tot+=1
                    if(ps!=-1):
                        count+=1
                        rows.add(i)
    print("Total number of cells: ", tot)
    print("Total number of errors: ", count)
    print('Rows involved: ', len(rows))
    perc=(count*100)/tot
    print("Percentage of error: ", perc)
    return rows

def correctMinusOne(data):
    for i in range(0, len(data)):
        for j in range(12, 17):
            ba=data.iat[i, j]
            if(ba>0):
                pa=data.iat[i, j+5]
                if((ba-pa)<=0):
                    ps=data.iat[i, j-6]
                    if(ps!=-1):
                        data.iloc[i,j-6]=-1
    return data

def findMinusTwo(data):
    tot=0
    count=0
    rows=set()
    for i in range(0, len(data)):
        for j in range(11,16):
            ba=data.iat[i,j]
            precBa=data.iat[i,j+1]
            pa=data.iat[i,j+6]
            ps=data.iat[i,j-6]
            if(ba<=0):
                if(ba==(precBa-pa)):
                    tot+=1
                    if(ps!=-2):
                        count+=1
                        rows.add(i)
    print("Total number of cells: ", tot)
    print("Total number of errors: ", count)
    print('Rows involved: ', len(rows))
    perc=(count*100)/tot
    print("Percentage of error: ", perc)
    return rows

def correctMinusTwo(data):
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
    return data


def data_correct(df_in):
    df_out1 = correctMinusOne(df_in)
    df_out2 = correctMinusTwo(df_out1)
    return df_out2


#REMOVE MISSING VALUES
def remove_missing_values(df_in):
    df_out = df_in
    df_out['sex'] = df_out['sex'].fillna(df_out['sex'].mode()[0])
    df_out['education'] = df_out['education'].fillna('others')
    df_out['status'] = df_out['status'].groupby([df_out['sex'], 
                                                 df_out['education']]).apply(
        lambda x: x.fillna(x.mode()[0]))
    df_out['age'] = df_out['age'].groupby([df_out['sex'], df_out['education'],
      df_out['status']]).apply(lambda x: x.replace(-1, x.median()))
    
    return df_out

#REMOVE OUTLIERS
def removeOutliers(dataFrame):
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
    print("Number of rows to be dropped: ", len(rows))
    dataFrame.drop(dataFrame.index[rows], inplace=True)
    print("size: ", len(dataFrame))
    
    return dataFrame


cc = pd.read_csv('C:\\Users\\Richard\\Desktop\\DM_proj\\credit_default_train.csv')
cc_correct = data_correct(cc)
cc_NoMV = remove_missing_values(cc_correct)
cc_final = removeOutliers(cc_NoMV)

cc_final.to_csv(path_or_buf='C:\\Users\\Richard\\Desktop\\DM_proj\\credit_default_FINAL.csv',index=False)
