"""
Validation of the model

"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np


class ValidationLoop():
    #validation of model using pytorch
    def __init__(self, model, dataset, user_item_mat, batch_size, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.user_item_mat = user_item_mat
        self.batch_size = batch_size
        self.device = device

    def __call__(self):
        print('Validation...')
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        loss = 0
        for batch_idx, (batch_data, batch_row_idx, batch_col_idx) in enumerate(dataloader):
            batch_data = Variable(batch_data).to(self.device)
            batch_row_idx = Variable(batch_row_idx).to(self.device)
            batch_col_idx = Variable(batch_col_idx).to(self.device)
            batch_pred = self.model(batch_row_idx, batch_col_idx)
            loss += torch.sum((batch_pred - batch_data)**2).item()
        loss /= len(self.dataset)
        return loss

    def predict(self, user_idx):
        self.model.eval()
        user_idx = Variable(torch.LongTensor([user_idx])).to(self.device)
        item_idx = Variable(torch.LongTensor(np.arange(self.user_item_mat.shape[1]))).to(self.device)
        return self.model(user_idx, item_idx).cpu().detach().numpy()

    def predict_all(self):
        self.model.eval()
        user_idx = Variable(torch.LongTensor(np.arange(self.user_item_mat.shape[0]))).to(self.device)
        item_idx = Variable(torch.LongTensor(np.arange(self.user_item_mat.shape[1]))).to(self.device)
        return self.model(user_idx, item_idx).cpu().detach().numpy()
    
        
  
    

