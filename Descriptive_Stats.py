#Import some relevant libraries
import pandas as pd
import numpy as np
import datetime as dt
from statistics import stdev, mean
from pathlib import Path

# Import data and create a dataframe

folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    df = pd.read_csv(file)

    df['k'] = ""

    

    for i in range(1, df.shape[0], 1):
        df.iloc[i, -1] = df.iloc[i-1, -2] - df.iloc[i, -2]

    df.to_csv(file, index = False)