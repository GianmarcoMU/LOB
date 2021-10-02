###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 03 - Deep Learning applied to LOB Data #
###########################################

# This code implements step 3 of the analysis (see pdf). Briefly, this means that it splits
# the dataset into train and test datasets for each instrument.

import pandas as pd
import numpy as np
import datetime as dt
from statistics import stdev, mean
from pathlib import Path

folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    df = pd.read_csv(file)

    name_file = str(file)[-9:-4]

    df['Date_Adj'] = ""

    format_date = "%Y-%m-%d"

    i = 0

    while i < len(df):
        df.iloc[i, -1] = dt.datetime.strptime(df.iloc[i, -2], format_date).date()
        i += 1

    df.reset_index()

    df_new = df.drop(columns = ['Date'])

    df_train = df_new.loc[df['Date_Adj'] < dt.datetime.strptime("2019-10-01", format_date).date()]

    df_test = df_new.loc[df['Date_Adj'] >= dt.datetime.strptime("2019-10-01", format_date).date()]
    
    df.reset_index()

    df_train.to_csv(r'C:\Users\mulaz\Desktop\DATA\Data_'+'Train_'+name_file+r'.csv', index = False)
    df_test.to_csv(r'C:\Users\mulaz\Desktop\DATA\Data_'+'Test_'+name_file+r'.csv', index = False)

