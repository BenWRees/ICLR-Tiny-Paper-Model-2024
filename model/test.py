"""
	Testing the Neural Networks
"""
import torch
from torch import nn, div, square, norm
from torch.nn import functional as F
from torchdata import datapipes as dp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import sys 

sys.path.insert(1,'/Users/benjaminrees/Desktop/ICLR-Tiny-Paper-Model-2024/Data_Processing')



from Autoencoders import *
from topological_regularisation import TopologicallyRegularizedAutoencoder
from training import TrainingLoop
from data_preprocessing import CollabFilteringPreProcessing


data_filter = CollabFilteringPreProcessing(['/Users/benjaminrees/Desktop/ml-1m/users.dat',
    '/Users/benjaminrees/Desktop/ml-1m/movies.dat', '/Users/benjaminrees/Desktop/ml-1m/ratings.dat'],12,512, 0.2)

num_users = data_filter.get_num_users()

model_one = AutoRec(d=num_users, k=500, dropout=0.01).to(torch.device('cpu'))
model_two = AutoRec_Regularisation(d=num_users, k=500, dropout=0.01, regularisation_param=0.001).to(torch.device('cpu'))

top_model_one = TopologicallyRegularizedAutoencoder(lam=0.01, autoencoder_model = 'AutoRec', 
    ae_kwargs={'k':500, 'd':num_users, 'dropout':0.01}, 
    toposig_kwargs = {'sort_selected':False, 'use_cycles':False, 'match_edges':None} )

max_epochs = 100

num_workers = 2

train_dp, test_dp = data_filter()
user_item_mat = data_filter.get_user_item_matrix()

#need to choose right dataset for this 
training_one = TrainingLoop(top_model_one, train_dp, user_item_mat, max_epochs, num_workers, 0.0001)
_, loss_one = training_one()
training_two= TrainingLoop(model_one, train_dp, user_item_mat, max_epochs, num_workers, 0.0001)
_, loss_two = training_two()


print('end loss TopoCF: ', loss_one[-1])
print('end loss AutoRec: ', loss_two[-1])


plt.title('training loss')
plt.plot(loss_one)
plt.plot(loss_two)
plt.legend(['TopoCF', 'AutoRec'])
plt.xlabel('epochs')
plt.ylabel('loss in MSE')
plt.show()


#Train AutoRec

#Train AutoRec_Regularisation


#Train TopologicallyRegularizedAutoencoder under AE
