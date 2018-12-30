import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fim import apriori

mydf = pd.read_csv(
    "../../../Dataset/credit_default_cleaned.csv", skipinitialspace=True)
