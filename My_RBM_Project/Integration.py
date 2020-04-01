# from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division
#exec('from __future__ import absolute_import, division, print_function')
# set the environment path to find Recommenders
import sys
sys.path.append("../../")
import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import papermill as pm

from reco_utils.recommender.rbm.rbm import RBM
from reco_utils.dataset.python_splitters import numpy_stratified_split
from reco_utils.dataset.sparse import AffinityMatrix


from reco_utils.dataset import movielens
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

electronics_data = pd.read_csv('SnackBars.csv')

# Convert to 32-bit in order to reduce memory consumption 
electronics_data.loc[:, 'Rating'] = electronics_data['Rating'].astype(np.int32)
electronics_data.loc[:, 'Price'] = electronics_data['Price'].astype(np.int32) 

header = {
        "col_user": "UserId",
        "col_item": "Snack Subscription ID",
        "col_rating": "Rating",
        #"col_rating": "Price",
    }

#instantiate the sparse matrix generation  
am = AffinityMatrix(DF = electronics_data, **header)

#obtain the sparse matrix 
X = am.gen_affinity_matrix()

#df_train.to_csv ('Trained_output.csv', index = False, header=True)

Xtr, Xtst = numpy_stratified_split(X)
selection = st.slider( 'Select a range of epoch',
    10, 100)
HiddenUnits = st.slider('Select the number of hidden layers', 100,600)

#First we initialize the model class
model = RBM(hidden_units= HiddenUnits, training_epoch = selection, minibatch_size= 60, keep_prob=0.9)

#Model Fit
train_time= model.fit(Xtr, Xtst)

#number of top score elements to be recommended  
K = 10

#Model prediction on the test set Xtst. 
top_k, test_time =  model.recommend_k_items(Xtst)

top_k_df = am.map_back_sparse(top_k, kind = 'prediction')
test_df = am.map_back_sparse(Xtst, kind = 'ratings')



#top_k_df = pd.dataframe(top_k_df)

st.dataframe(top_k_df)







