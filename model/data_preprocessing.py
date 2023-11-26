"""
	Functions for processing user and item data into a user-item matrix to be read by the autoencoder
	Method developed from https://github.com/tuanio/AutoRec as an adaptation to Sedhain, Suvash, et al. 
	"Autorec: Autoencoders meet collaborative filtering." Proceedings of the 24th international conference on World Wide Web. 2015.
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



class CollabFilteringPreProcessing :

	def __init__(self, datapaths, seed=12, batch_size=512, split_test_percent=0.2) :
		self.datapaths = datapaths
		self.seed = seed
		self.batch_size = batch_size
		self.split_test_percent = split_test_percent
		self.users_dat, self.items_dat, self.ratings = self.read_csvs()
		print('user data: \n', self.users_dat)
		print('items data: \n', self.items_dat)
		print('ratings data: \n', self.ratings)

	def __call__(self) :

		train_items, test_items = self.split_data(self.items_dat[0].max())
		self.user_item_mat = torch.zeros((self.get_num_users(), self.get_num_items()))
		self.user_item_mat = self.populate_user_item_matrix(self.ratings, self.user_item_mat)

		print('user item matrix: \n', self.user_item_mat)
	

		train_dp = self.create_datapipe_from_array(train_items, 'train', self.batch_size)
		test_dp = self.create_datapipe_from_array(test_items, 'test', self.batch_size)

		return (train_dp, test_dp)

	def get_num_users(self) :
		return self.users_dat[0].max()

	def get_num_items(self) :
		return self.items_dat[0].max()	


	def get_user_item_matrix(self) :
		return self.user_item_mat 

	def read_csvs(self) :
		users_dat = pd.read_csv(self.datapaths[0],
            delimiter='::',
            engine='python',
            encoding='latin-1',
            header=None)
		items_dat = pd.read_csv(self.datapaths[1],
            delimiter='::',
            engine='python',
            encoding='latin-1',
            header=None)

		ratings = pd.read_csv(self.datapaths[2],
            encoding='latin-1',
            header=None,
            engine='python',
            delimiter='::')
		return users_dat, items_dat, ratings

	def split_data(self,num_items) :
		train_items, test_items = train_test_split(torch.arange(num_items),test_size=self.split_test_percent,random_state=self.seed)
		return train_items, test_items

	def create_data_from_line(self,line, user_item_mat):
		user_id, item_id, rating, *_ = line
		self.user_item_mat[user_id - 1, item_id - 1] = rating
		return None


	def populate_user_item_matrix(self, ratings, user_item_mat) :
		application_fn = lambda x : self.create_data_from_line(x,user_item_mat)
		ratings.T.apply(application_fn); #populates the user_item_matrix
		return self.user_item_mat

	def collate_fn(self, batch):
		return torch.LongTensor(batch)

	def create_datapipe_from_array(self,array, mode, batch_size, len=1000) :
			

		pipes = dp.iter.IterableWrapper(array)
		pipes = pipes.shuffle(buffer_size=len)
		pipes = pipes.sharding_filter()

		if mode == 'train':
			pipes = pipes.batch(batch_size, drop_last=True)
		else:
			pipes = pipes.batch(batch_size)

		pipes = pipes.map(self.collate_fn)
		return pipes











