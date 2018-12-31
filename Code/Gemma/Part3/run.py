import pandas as pd
#import sys
#sys.path.insert(0, '../Gemma/Part3')
from pMining import binCols
from pMining import remapCols
from pMining import sortCols
from pMining import allFreqPatterns
from pMining import closedFreqPatterns
from pMining import minimalFreqPatterns
from pMining import associationRules
from pMining import dataVisual
mydf = pd.read_csv(
    "../../../Dataset/credit_default_cleaned.csv", skipinitialspace=True)
print("letto csv")
binCols(mydf)
print("fatto binCols")
remapCols(mydf)
print("fatto remap")
sortCols(mydf)
print("sortate colonne")
baskets = mydf.values.tolist()
print("creati basket")
allFreqPatterns(baskets, 60)
closedFreqPatterns(baskets, 60)
minimalFreqPatterns(baskets, 60)
associationRules(baskets, 60)
#dataVisual(60)
