###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 01 - Deep Learning applied to LOB Data #
###########################################

# This code implements step 1 of the analysis (see pdf). Briefly, this means that it creates
# a separate csv file for each instrument. Final output is made by 5 csv files.

import pandas as pd
from pathlib import Path

folder = r"C:\Users\mulaz\Desktop\DATA"

for file in Path(folder).glob('*.csv'):
    
    data = pd.read_csv(file)

    data_category_range = data['#RIC'].unique()
    data_category_range = data_category_range.tolist()

    for i,value in enumerate(data_category_range):
        data[data['#RIC'] == value].to_csv(r'C:\Users\mulaz\Desktop\DATA\Data_'+str(value)+r'.csv',index = False, na_rep = 'N/A')