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
    #df = df.drop(columns = ['Time', 'Date'])

    df['Mid-Price'] = (df['L1-AskPrice'] + df['L1-BidPrice'])/2

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

    for k in range(21, df.shape[0]-20, 1):

        k1_prev_mid_prices['pr_price{r}'.format(r=k)] = []
        k1_next_mid_prices['next_price{r}'.format(r=k)] = []
        k2_prev_mid_prices['pr_price{r}'.format(r=k)] = []
        k2_next_mid_prices['next_price{r}'.format(r=k)] = []
        k3_prev_mid_prices['pr_price{r}'.format(r=k)] = []
        k3_next_mid_prices['next_price{r}'.format(r=k)] = []

        mid_prices_means['k1_pr_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k1_next_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k2_pr_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k2_next_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k3_pr_price_mean{r}'.format(r=k)] = []
        mid_prices_means['k3_next_price_mean{r}'.format(r=k)] = []

    for i in range(21, df.shape[0]-20, 1):
        for x in range(1, 21, 1):             
            k1_prev_mid_prices['pr_price{r}'.format(r=i)].append(df.iloc[i-x, -1])
            k1_next_mid_prices['next_price{r}'.format(r=i)].append(df.iloc[i+x, -1])

    print(k1_prev_mid_prices)

    df.to_csv(file, index = False)