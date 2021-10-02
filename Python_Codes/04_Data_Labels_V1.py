###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 03 - Deep Learning applied to LOB Data #
###########################################

# This code implements step 4 of the analysis (see pdf). Briefly, this means that it computes
# labels associated to each state of the LOB, and they indicate if the price is upward (1), 
# downward (2) or steady (3). Here, the method used is the one by Zhang et al.

#Import some relevant libraries
import pandas as pd
import numpy as np
import datetime as dt
from statistics import stdev, mean
from pathlib import Path

folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    df = pd.read_csv(file)
    #df = df.drop(columns = ['Time', 'Date'])

    df['Mid-Price'] = (df['L1-AskPrice'] + df['L1-BidPrice'])/2

    #print(df.head(5))
    #print(df.shape[0])

    # k1 = 20
    k1_prev_mid_prices = {}
    k1_next_mid_prices = {}

    # k2 = 50
    k2_prev_mid_prices = {}
    k2_next_mid_prices = {}

    # k3 = 100
    k3_prev_mid_prices = {}
    k3_next_mid_prices = {}

    mid_prices_means = {}

    for k in range(20, df.shape[0]-20, 1):

        k1_prev_mid_prices['pr_price{r}'.format(r=k)] = []
        k1_next_mid_prices['next_price{r}'.format(r=k)] = []
        mid_prices_means['k1_pr_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k1_next_price_mean{r}'.format(r=k)] = []

    for k in range(50, df.shape[0]-50, 1):

        k2_prev_mid_prices['pr_price{r}'.format(r=k)] = []
        k2_next_mid_prices['next_price{r}'.format(r=k)] = []
        mid_prices_means['k2_pr_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k2_next_price_mean{r}'.format(r=k)] = []

    for k in range(100, df.shape[0]-100, 1):

        k3_prev_mid_prices['pr_price{r}'.format(r=k)] = []
        k3_next_mid_prices['next_price{r}'.format(r=k)] = []      
        mid_prices_means['k3_pr_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k3_next_price_mean{r}'.format(r=k)] = []

    for i in range(20, df.shape[0]-20, 1):
        for x in range(0, 21, 1):             
            k1_prev_mid_prices['pr_price{r}'.format(r=i)].append(df.iloc[i-x, -1])
       
        for x in range(1, 21, 1):
            k1_next_mid_prices['next_price{r}'.format(r=i)].append(df.iloc[i+x, -1])
        
        mid_prices_means['k1_pr_price_mean{r}'.format(r=i)] = np.mean(k1_prev_mid_prices['pr_price{r}'.format(r=i)])
        mid_prices_means['k1_next_price_mean{r}'.format(r=i)] = np.mean(k1_next_mid_prices['next_price{r}'.format(r=i)])

    for i in range(50, df.shape[0]-50, 1):
        for x in range(0, 51, 1):             
            k2_prev_mid_prices['pr_price{r}'.format(r=i)].append(df.iloc[i-x, -1])
       
        for x in range(1, 51, 1):
            k2_next_mid_prices['next_price{r}'.format(r=i)].append(df.iloc[i+x, -1])

        mid_prices_means['k2_pr_price_mean{r}'.format(r=i)] = np.mean(k2_prev_mid_prices['pr_price{r}'.format(r=i)])
        mid_prices_means['k2_next_price_mean{r}'.format(r=i)] = np.mean(k2_next_mid_prices['next_price{r}'.format(r=i)])

    for i in range(100, df.shape[0]-100, 1):
        for x in range(0, 101, 1):             
            k3_prev_mid_prices['pr_price{r}'.format(r=i)].append(df.iloc[i-x, -1])
       
        for x in range(1, 101, 1):
            k3_next_mid_prices['next_price{r}'.format(r=i)].append(df.iloc[i+x, -1])

        mid_prices_means['k3_pr_price_mean{r}'.format(r=i)] = np.mean(k3_prev_mid_prices['pr_price{r}'.format(r=i)])
        mid_prices_means['k3_next_price_mean{r}'.format(r=i)] = np.mean(k3_next_mid_prices['next_price{r}'.format(r=i)])

    df['Label_20'] = ""
    df['Label_50'] = ""
    df['Label_100'] = ""

    alpha = 0.002

    for i in range(20, df.shape[0]-20, 1):
        if (mid_prices_means['k1_next_price_mean{r}'.format(r=i)] - mid_prices_means['k1_pr_price_mean{r}'.format(r=i)])/mid_prices_means['k1_pr_price_mean{r}'.format(r=i)] >= alpha:
            df.iloc[i, -3] = 1
        if (mid_prices_means['k1_next_price_mean{r}'.format(r=i)] - mid_prices_means['k1_pr_price_mean{r}'.format(r=i)])/mid_prices_means['k1_pr_price_mean{r}'.format(r=i)] <= -alpha:
            df.iloc[i,-3] = 2
        else:
            df.iloc[i,-3] = 3

    for i in range(50, df.shape[0]-50, 1):
        if (mid_prices_means['k2_next_price_mean{r}'.format(r=i)] - mid_prices_means['k2_pr_price_mean{r}'.format(r=i)])/mid_prices_means['k2_pr_price_mean{r}'.format(r=i)] >= alpha:
            df.iloc[i, -2] = 1
        if (mid_prices_means['k2_next_price_mean{r}'.format(r=i)] - mid_prices_means['k2_pr_price_mean{r}'.format(r=i)])/mid_prices_means['k2_pr_price_mean{r}'.format(r=i)] <= -alpha:
            df.iloc[i,-2] = 2
        else:
            df.iloc[i, -2] = 3

    for i in range(100, df.shape[0]-100, 1):
        if (mid_prices_means['k3_next_price_mean{r}'.format(r=i)] - mid_prices_means['k3_pr_price_mean{r}'.format(r=i)])/mid_prices_means['k3_pr_price_mean{r}'.format(r=i)] >= alpha:
            df.iloc[i, -1] = 1
        if (mid_prices_means['k3_next_price_mean{r}'.format(r=i)] - mid_prices_means['k3_pr_price_mean{r}'.format(r=i)])/mid_prices_means['k3_pr_price_mean{r}'.format(r=i)] <= -alpha:
            df.iloc[i, -1] = 2
        else:
            df.iloc[i, -1] = 3

    df.to_csv(file, index = False)