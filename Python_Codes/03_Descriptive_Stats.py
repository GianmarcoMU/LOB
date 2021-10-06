###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 03 - Deep Learning applied to LOB Data #
###########################################

# This code is used to compute some basic descriptive statistics about our data.
# In particular, we count the average number of LOB states in a day, the average interval
# between two consecutive states. Plus, the total number of states in the 12 months.

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

    daily_obs_count = df.groupby('Date').agg({'Date':['count']})

    print(daily_obs_count)
    print(np.mean(daily_obs_count))
    print(np.sum(daily_obs_count))

    df['Time'] = ""
    df['Time_Adj'] = ""

    i = 0

    while i < len(df):
        df.iloc[i, -2] = (df.iloc[i, 1][11:-1])
        df.iloc[i, -1] = dt.datetime.strptime(df.iloc[i, -2][0:-3], "%H:%M:%S.%f")
        i += 1

    state_change = {'state_change_val' : []}

    for k in range(1, len(df), 1):
        diff = df.iloc[k,-1]-df.iloc[k-1,-1]
        state_change['state_change_val'].append(diff)

    print(np.mean(state_change['state_change_val']))

