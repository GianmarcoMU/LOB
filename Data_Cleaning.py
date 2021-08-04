#Import some relevant libraries
import pandas as pd
import numpy as np

#Import and read a CSV file
#We add the 'r' before the path specification to solve an error that occured otherwise
file_path = r"C:\Users\mulaz\Desktop\GIANMARCO\BOCCONI\MSc - ESS\A.Y. 2020-21\SPRING SEMESTER\FINAL THESIS\DATA\TRTH_L2\Mulazzani\split\Market Depth NYSE_chunk1.csv"
df = pd.read_csv(file_path)
#print(chunck1['Date-Time'].head())

# The idea of the first step is to separate date and time in two different columns
# then we need to normalise the hour and so subtract 4, then delete the useless columns
# this should be done for all csv file and all rows, thus we need a for loop

# This code is for extracting Date-Time column, select the hour and subtract GMT offset
# It does not work
slice = df.iloc[:,2]
print(slice[11:13])

#print(int(slice[11:13])-4)
