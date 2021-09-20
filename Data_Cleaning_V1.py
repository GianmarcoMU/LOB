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
    df = df.drop(columns = ['Domain', 'GMT Offset', 'Type'])

    format = "%H:%M:%S"

    opening = "13:30:00"
    closing = "20:00:00"
    check_o = dt.datetime.strptime(opening, format).time()
    check_c = dt.datetime.strptime(closing, format).time()

    df['Date'] = df['Date-Time'].str[:10]
    df['Time'] = df['Date-Time'].str[11:-1]
    df['Time_Adj'] = ""
    df['Check'] = ""

    i = 0

    while i < len(df):
        df.iloc[i, -2] = dt.datetime.strptime(df.iloc[i, -3][0:8], format).time()
        df.iloc[i, -1] = (df.iloc[i,-2] < check_o) or (df.iloc[i,-2] > check_c)
        i += 1

    df = df.loc[df['Check'] == False]

    df_new = df.drop(columns = ['Date-Time', 'Time_Adj', 'Check'])
    df_new.to_csv(file, index = False)


