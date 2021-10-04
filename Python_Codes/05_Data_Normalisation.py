###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 03 - Deep Learning applied to LOB Data #
###########################################

# This code implements step 5 of the analysis (see pdf). Briefly, this means that it is used
# to normalise data according to the procedure described in Zhang et al.

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
    df = df.drop(columns = ['#RIC', 'Date_Adj', 'Mid-Price'])

    name_file = str(file)[5:-1]
    
    #print(df.head(5))

    five_previous = {}
    five_previous_stats = {}

    for k in range(df.shape[0]-5):

        five_previous['price{r}'.format(r=k)] = []
        five_previous['vol{r}'.format(r=k)] = []
        five_previous_stats['price_mean{r}'.format(r=k)] = []
        five_previous_stats['price_stdev{r}'.format(r=k)] = []
        five_previous_stats['vol_mean{r}'.format(r=k)] = []
        five_previous_stats['vol_stdev{r}'.format(r=k)] = []   
    
    for i in range(df.shape[0]-5):
        for z in range(0, df.shape[1]-3, 2):
            for x in range(1, 6, 1):                
                five_previous['price{r}'.format(r=i)].append(df.iloc[5+i-x, z])
                five_previous['vol{r}'.format(r=i)].append(df.iloc[5+i-x, z+1])

        five_previous_stats['price_mean{r}'.format(r=i)] = np.mean(five_previous['price{r}'.format(r=i)])
        five_previous_stats['price_stdev{r}'.format(r=i)] = np.std(five_previous['price{r}'.format(r=i)])
        five_previous_stats['vol_mean{r}'.format(r=i)] = np.mean(five_previous['vol{r}'.format(r=i)])
        five_previous_stats['vol_stdev{r}'.format(r=i)] = np.std(five_previous['vol{r}'.format(r=i)])               
            
    #print(five_previous_stats)

    for p in range(0, df.shape[0]-5, 1):
        for h in range(0, df.shape[1]-3, 2):
            df.iloc[5+p, h] = (df.iloc[5+p, h] - five_previous_stats['price_mean{r}'.format(r=p)])/five_previous_stats['price_stdev{r}'.format(r=p)]
            df.iloc[5+p, h+1] = (df.iloc[5+p, h+1] - five_previous_stats['vol_mean{r}'.format(r=p)])/five_previous_stats['vol_stdev{r}'.format(r=p)]
    
    
    df = df.drop([0,1,2,3,4])
  
    df.to_csv(file, index = False)

    # Since other codes use np and txt format, a way to save it in txt is:
    # df.to_csv(r'C:\Users\mulaz\Desktop\DATA\Data_'+'Normalised_'+name_file+r'.txt', sep='\t', index=False)