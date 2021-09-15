## Gianmarco Mulazzani - 2021 ##

## This code is intended for "experimenting" some data cleaning and normalization of 
## a sample dataset taken from the final dataset.

# From the DeepLOB paper, authors state that they use the mean and st.deviation of the previous
# five days to normalize data of each day. But they we discard the first 5 obs?

# Note that NYSE opening continuous trading hours are: 9.30 - 16.00
# This means that we need to delete obs before 13.30 and after 20.00

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

    format = "%H:%M:%S"

    opening = "13:30:00"
    closing = "16:00:00"
    check_o = dt.datetime.strptime(opening, format).time()
    check_c = dt.datetime.strptime(closing, format).time()

    df['Date'] = df['Date-Time'].str[:10]
    df['Time'] = df['Date-Time'].str[11:-1]
    df['Time_Adj'] = ""
    df['Check'] = ""

    for i in range(0,20):
        df.iloc[i, -2] = dt.datetime.strptime(df.iloc[i, -3][0:8], format).time()
        df.iloc[i, -1] = (df.iloc[i,-2] < check_o) or (df.iloc[i,-2] > check_c)

    df.drop(df.loc[df['Check']==True].index, inplace=True)   
    df.reset_index(drop=True, inplace=True)

    print(df.head(10))


#print(df.head(5)) # Print the first five rows
#print(df.columns) # Print the name of the columns
#print(df.iloc[1, 2]) # Print element in row 2 and column 3

# Create a for loop to retrieve all the rows for a specific column
#for index, row in df.iterrows():
 #   print(row[2])

#print(type(df.iloc[1, 2])) # Here we check that the content of Date-Time is a string

# Here we create two new columns to separate Date and Time



## Now that we have cleaned the dataset, we need to normalise it. In order to do so, we need to
## compute mean and std.dev for both prices and volumes using the previous five days at any given point.

## An idea could be to initialize empty dictionaries and then populate them with the obs of previous days. This could be done by using while loops
## check the agenda for more details. 

five_previous_prices = list()
five_previous_volumes = list()

z = 5
h = 5

while z < 45:
    for x in range(1, 6, 1):
        five_previous_prices.append(df.iloc[5-x, z])
        five_previous_volumes.append(df.iloc[5-x, z+1])
    z += 2


while h < 45:
    df.iloc[5, h] = (df.iloc[5, h] - mean(five_previous_prices))/stdev(five_previous_prices)
    h += 2

print(df.iloc[5,5])







