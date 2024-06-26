# GinAR
This github repository corresponds to our paper accepted by SIGKDD 2024 (GinAR: An End-To-End Multivariate Time Series Forecasting Model Suitable for Variable Missing).

All datasets can be found in this link: https://github.com/ChengqingYu/MTS_dataset

The main difference between Model 1 and Model 2 lies in the way they use interpolation attention and the composition of weights in the graph convolution within the GinAR cell.

The code for data processing and model training is currently being organized (Since I am currently preparing for the defense of my doctoral proposal report, I will try my best to organize and upload the code in July after completing these matters.). 

The setting of missing values is introduced in Section 3.1 (Preliminaries) of our paper. We randomly generate M numbers proportionally, and for the input feature X (which consists of N time series, with M being smaller than N), we convert the values of the corresponding M variables among the N variables to zero. Setting missing variables to zero is based on methods discussed in time series imputation-related papers such as GRIN.


The following is the meaning of the core hyperparameter:
- input_len: The length of historical observation 
- num_id: The number of variables
- out_len: The length of forecasting steps 
- in_size:  The number of input features (Details you can refer to: https://github.com/zezhishao/BasicTS)
- emb_size: Embedding size
- grap_size: Variable embedding size
- layer_num: The number of GinAR layer
- dropout: dropout
- adj_mx: Adjacency matrix. (Details you can refer to: https://github.com/zezhishao/BasicTS)

