###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 05 - Deep Learning applied to LOB Data #
###########################################

# This code implements step 5 of the analysis. Briefly, this means that it computes
# labels associated to each state of the LOB, and they indicate if the price is upward (1), 
# downward (3) or steady (2). Here, the method used is the one by Ntakaris, Magris et al.
# For more details see their paper. Note that this is the same method used to construct the
# benchmark dataset FI-2010.

#Import some relevant libraries
import pandas as pd
import numpy as np
import datetime as dt
from statistics import stdev, mean
from pathlib import Path

# Change the directory appropriately.
folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    df = pd.read_csv(file)
    
    df['Mid-Price'] = (df['L1-AskPrice'] + df['L1-BidPrice'])/2

    # The idea is to build auxiliary dictionaries where to store values needed.

    # k1 = 20
    k1_next_mid_prices = {}

    # k2 = 50
    k2_next_mid_prices = {}

    # k3 = 100
    k3_next_mid_prices = {}

    mid_prices_means = {}

    for k in range(0, df.shape[0]-20, 1):

        k1_next_mid_prices['next_price{r}'.format(r=k)] = []
        mid_prices_means['k1_next_price_mean{r}'.format(r=k)] = []

    for k in range(0, df.shape[0]-50, 1):

        k2_next_mid_prices['next_price{r}'.format(r=k)] = []
        mid_prices_means['k2_next_price_mean{r}'.format(r=k)] = []

    for k in range(0, df.shape[0]-100, 1):

        k3_next_mid_prices['next_price{r}'.format(r=k)] = []      
        mid_prices_means['k3_next_price_mean{r}'.format(r=k)] = []

    for i in range(0, df.shape[0]-20, 1):     
        for x in range(1, 21, 1):
            k1_next_mid_prices['next_price{r}'.format(r=i)].append(df.iloc[i+x, -1])
        
        mid_prices_means['k1_next_price_mean{r}'.format(r=i)] = np.mean(k1_next_mid_prices['next_price{r}'.format(r=i)])

    for i in range(0, df.shape[0]-50, 1):
        for x in range(1, 51, 1):
            k2_next_mid_prices['next_price{r}'.format(r=i)].append(df.iloc[i+x, -1])

        mid_prices_means['k2_next_price_mean{r}'.format(r=i)] = np.mean(k2_next_mid_prices['next_price{r}'.format(r=i)])

    for i in range(0, df.shape[0]-100, 1):
        for x in range(1, 101, 1):
            k3_next_mid_prices['next_price{r}'.format(r=i)].append(df.iloc[i+x, -1])

        mid_prices_means['k3_next_price_mean{r}'.format(r=i)] = np.mean(k3_next_mid_prices['next_price{r}'.format(r=i)])


    df['Label_20'] = ""
    df['Label_50'] = ""
    df['Label_100'] = ""

    alpha = 0.002 # This value is the same used by Ntakaris et al. It could be change if necessary.

    for i in range(0, df.shape[0]-20, 1):
        if (mid_prices_means['k1_next_price_mean{r}'.format(r=i)] - df.iloc[i,-4])/df.iloc[i,-4] >= alpha:
            df.iloc[i, -3] = 1
        elif (mid_prices_means['k1_next_price_mean{r}'.format(r=i)] - df.iloc[i,-4])/df.iloc[i,-4] <= -alpha:
            df.iloc[i,-3] = 3
        else:
            df.iloc[i,-3] = 2

    for i in range(0, df.shape[0]-50, 1):
        if (mid_prices_means['k2_next_price_mean{r}'.format(r=i)] - df.iloc[i,-4])/df.iloc[i,-4] >= alpha:
            df.iloc[i, -2] = 1
        elif (mid_prices_means['k2_next_price_mean{r}'.format(r=i)] - df.iloc[i,-4])/df.iloc[i,-4] <= -alpha:
            df.iloc[i,-2] = 3
        else:
            df.iloc[i, -2] = 2

    for i in range(0, df.shape[0]-100, 1):
        if (mid_prices_means['k3_next_price_mean{r}'.format(r=i)] - df.iloc[i,-4])/df.iloc[i,-4] >= alpha:
            df.iloc[i, -1] = 1
        elif (mid_prices_means['k3_next_price_mean{r}'.format(r=i)] - df.iloc[i,-4])/df.iloc[i,-4] <= -alpha:
            df.iloc[i, -1] = 3
        else:
            df.iloc[i, -1] = 2

    df = df.iloc[:-100] # Here we drop the last 100 rows for which labels cannot be associated.
    df.reset_index()

    df.to_csv(file, index = False)