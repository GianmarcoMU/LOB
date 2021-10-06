###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 02 - Deep Learning applied to LOB Data #
###########################################

# This code implements step 2 of the analysis. Briefly, this means that it cleans
# data by eliminating LOB states outside normal trading hours (9:30 - 16:00 for NYSE).

import pandas as pd
import numpy as np
import datetime as dt
from statistics import stdev, mean
from pathlib import Path

# Change the directory appropriately.
folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    df = pd.read_csv(file)
    
    format_time = "%H:%M:%S"

    # Note that the distinction between GMT-4 and GMT-5 is needed due to the time change in Europe.

    opening_GMT4 = "13:30:00"
    closing_GMT4 = "20:00:00"
    check_o_GMT4 = dt.datetime.strptime(opening_GMT4, format_time).time()
    check_c_GMT4 = dt.datetime.strptime(closing_GMT4, format_time).time()

    opening_GMT5 = "14:30:00"
    closing_GMT5 = "21:00:00"
    check_o_GMT5 = dt.datetime.strptime(opening_GMT5, format_time).time()
    check_c_GMT5 = dt.datetime.strptime(closing_GMT5, format_time).time()

    df['Date'] = df['Date-Time'].str[:10]
    df['Time'] = df['Date-Time'].str[11:-1]
    df['Time_Adj'] = ""
    df['Check'] = ""

    i = 0

    while i < len(df):
        df.iloc[i, -2] = dt.datetime.strptime(df.iloc[i, -3][0:8], format_time).time()

        if df.iloc[i, 3] == -4:
            df.iloc[i, -1] = (df.iloc[i,-2] < check_o_GMT4) or (df.iloc[i,-2] > check_c_GMT4)
        else:
            df.iloc[i, -1] = (df.iloc[i,-2] < check_o_GMT5) or (df.iloc[i,-2] > check_c_GMT5)

        i += 1

    df = df.loc[df['Check'] == False]
    df.reset_index()

    df_new = df.drop(columns = ['Domain', 'GMT Offset', 'Type', 'Time_Adj', 'Check', 'Time'])

    df_new.to_csv(file, index = False) # Clean dataset replaces the original one.