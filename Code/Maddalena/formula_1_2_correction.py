#Input: dataframe of the loaded credit cards dataset with ALL standard columns
#Output: dataframe with the corrected values in ps of the credit cards dataset
def correct_ps_values(data):
    #Correzione FORMULA 1
    #Sistemo valori -1
    for i in range(0, len(data)):
        for j in range(12, 17):correct_ps_values
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