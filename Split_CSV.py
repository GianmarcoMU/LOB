## This code is used to split the csv file and create a csv file for each stock. This is necessary since 
## normalization must be done for each instrument separately.

import pandas as pd

data = pd.read_csv('NYSE.csv')

data_category_range = data['#RIC'].unique()
data_category_range = data_category_range.tolist()

for i,value in enumerate(data_category_range):
    data[data['#RIC'] == value].to_csv(r'#RIC_'+str(value)+r'.csv',index = False, na_rep = 'N/A')