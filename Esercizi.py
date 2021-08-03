#Import some relevant libraries

import math as mt 
import datetime as dt
import pandas as pd

#Define just some variables in Python

a = 5
b = 'prova'
c = 3/4

list1 = [1, 2, 3, 4, 5]
list2 = [0, 1, 0, 1, 0]
list3 = [2, 2, 1, 1, 5]

list4 = [list1, list2, list3]
print(list4)

#Some exercises with the libraries

data = pd.DataFrame({'Lista 1': list1, 'Lista 2': list2, 'Lista 3': list3})
print(data)

#Import and read a CSV file
#We add the 'r' before the path specification to solve an error that occured otherwise
file_path = r"C:\Users\mulaz\Desktop\GIANMARCO\BOCCONI\MSc - ESS\A.Y. 2020-21\SPRING SEMESTER\FINAL THESIS\DATA\TRTH_L2\Mulazzani\split\Market Depth NYSE_chunk1.csv"
chunck1 = pd.read_csv(file_path)
print(chunck1.head())