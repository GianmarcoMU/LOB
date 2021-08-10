#Import some relevant libraries
import pandas as pd
import numpy as np

# Import data and create a dataframe
df = pd.read_csv('prova.csv')
print(df.head(5)) # Print the first five rows
print(df.columns) # Print the name of the columns
print(df.iloc[1, 2]) # Print element in row 2 and column 3

# Create a for loop to retrieve all the rows for a specific column
for index, row in df.iterrows():
    print(row[2])

print(type(df.iloc[1, 2])) # Here we check that the content of Date-Time is a string

# Here we create two new columns to separate Date and Time
df['Date'] = df['Date-Time'].str[:10]
df['Time'] = df['Date-Time'].str[11:-1]
print(df.head(5))

print(type(df.iloc[1, -1]))
