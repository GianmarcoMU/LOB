#Import some relevant libraries
import pandas as pd
import numpy as np
import datetime as dt
from statistics import stdev, mean
from pathlib import Path

## The aim of this script is to create the labels for our data that will constitute the outputs
## Firstly, we compute the mid-prices
## Then, we compute different mean of the k previous  and k next mid-prices. For k = 20, 50, 100
## Finally, l_t is computed and compared to the threshold alpha (0.002 in the paper)

# Import data and create a dataframe

folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    df = pd.read_csv(file)
    df = df.drop(columns = ['Domain', 'GMT Offset', 'Type'])

    df['Mid-Price'] = (df['L1-AskPrice'] + df['L1-BidPrice'])/2

    print(df.head(10))