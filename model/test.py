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

sys.path.insert(1,'/Users/aiden/OneDrive/Documents/GitHub/ICLR-Tiny-Paper-Model-2024/Data_Processing')



#from AutoRec import *
from Autoencoders import *
from topological_regularisation import TopologicallyRegularizedAutoencoder
from training import TrainingLoop
from validation import ValidationLoop
#from visualisations import *
from data_preprocessing import CollabFilteringPreProcessing


data_filter = CollabFilteringPreProcessing(['/Users/aiden/OneDrive/Documents/GitHub/ICLR-Tiny-Paper-Model-2024/ml-1m/users.dat',
    '/Users/aiden/OneDrive/Documents/GitHub/ICLR-Tiny-Paper-Model-2024/ml-1m/movies.dat', '/Users/aiden/OneDrive/Documents/GitHub/ICLR-Tiny-Paper-Model-2024/ml-1m/ratings.dat'],12,512)

num_users = data_filter.get_num_users()

model_one = AutoRec(d=num_users, k=500, dropout=0.01).to(torch.device('cpu'))
model_two = AutoRec_Regularisation(d=num_users, k=500, dropout=0.01, regularisation_param=0.0001).to(torch.device('cpu'))

# top_model_one = TopologicallyRegularizedAutoencoder(lam=0.01, autoencoder_model = 'AutoRec', 
#     ae_kwargs={'k':500, 'd':num_users, 'dropout':0.01}, 
#     toposig_kwargs = {'sort_selected':False, 'use_cycles':False, 'match_edges':None} )

max_epochs = 100

num_workers = 2

train_dp, test_dp = data_filter()
user_item_mat = data_filter.get_user_item_matrix()

#need to choose right dataset for this 
training_one = TrainingLoop(model_one, train_dp, user_item_mat, max_epochs, num_workers, 0.001)
_, loss_one = training_one()

training_two = TrainingLoop(model_two, train_dp, user_item_mat, max_epochs, num_workers, 0.0001)
_, loss_two = training_one()

testing_one = ValidationLoop(model_one, test_dp, user_item_mat, num_workers)
_, test_loss_one = testing_one()
testing_two = ValidationLoop(model_two, test_dp, user_item_mat, num_workers)
_, test_loss_two = testing_two()

print('end loss one: ', loss_one[-1])
print('end loss two: ', loss_two[-1])

"""
plt.title('training loss')
plt.plot(loss_one)
plt.plot(loss_two)
plt.legend(['regularisation=0.001',
            'regularisation=0.0001'])
plt.xlabel('epochs')
plt.ylabel('loss in MSE')
plt.show()
"""

plt.title('test loss')
plt.plot(test_loss_one)
plt.plot(test_loss_two)
plt.xlabel('epochs')
plt.ylabel('loss in MSE')

#Train AutoRec

#Train AutoRec_Regularisation


#Train TopologicallyRegularizedAutoencoder under AE
