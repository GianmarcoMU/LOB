# LOB
In this repository some codes related to the analysis of LOB data are included. In particular, the model DeepLOB (see the attached paper) is replicated to study NYSE data.

Data regards five stocks traded on NYSE between September 2019 and August 2020 such that the pandemic crisis is included. Moreover, the pre-pandemic period will be used as train set while the post-pandemic will be used as testing period. 

Codes included are: Data_Labels which perfomrs the computations of labels used as outputs in the analysis, Data_Cleaning that cleans data by removing the state of the LOB outside the normal trading hours, Data_Normalization which applies z-score normalisation methodology and finally the CNN-RNN_Model is just the pasted code implemented by Zhang and Zohren.
