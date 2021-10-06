###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 04 - Deep Learning applied to LOB Data #
###########################################

# This code implements step 4 of the analysis. Briefly, this means that it splits
# the dataset into train and test datasets for each instrument. Thus, we will obtain 10
# csv files, two for each stock. They will be merged only after normalisation (step 5).

import pandas as pd
import numpy as np
import datetime as dt
from statistics import stdev, mean
from pathlib import Path

# Change the directory appropriately.
folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    df = pd.read_csv(file)
    df.drop(columns = ['Date-Time'])

    name_file = str(file)[-9:-4]

    df['Date_Adj'] = ""

    format_date = "%Y-%m-%d"

    i = 0

    while i < len(df):
        df.iloc[i, -1] = dt.datetime.strptime(df.iloc[i, -2], format_date).date()
        i += 1

    df.reset_index()

    df_new = df.drop(columns = ['Date'])

    # Here, you can change the date based on how you want to split the dataset into train and test.

    df_train = df_new.loc[df['Date_Adj'] < dt.datetime.strptime("2020-01-31", format_date).date()]

    df_test = df_new.loc[df['Date_Adj'] >= dt.datetime.strptime("2020-01-31", format_date).date()]
    
    df.reset_index()

    df_train.to_csv(r'C:\Users\mulaz\Desktop\DATA\Data_'+'Train_'+name_file+r'.csv', index = False)
    df_test.to_csv(r'C:\Users\mulaz\Desktop\DATA\Data_'+'Test_'+name_file+r'.csv', index = False)

