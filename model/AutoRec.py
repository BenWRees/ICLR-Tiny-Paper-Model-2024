import numpy as np
import torch
import torch.nn as nn 
import abd

from Base Autencoder import AutoencoderModel


#The U-Autocode Model without regularisation 
class AutoRec(AutoencoderModel) :
	
	def __init__(self, k, d, dropout) :
		super().__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=d, out_features=k, bias=True),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )

		self.decoder = nn.Sequential(
			nn.Linear(in_features=k, out_features=d, bias=True)
            nn.ReLU(True),
            nn.Dropout(dropout),
     	)

		self.reconst_error = nn.MSELoss()

	def encode(self, x) :
		self.encoder(x)

	def decode(self, x) :
		self.decoder(x)

    def forward(self, x):
	    latent = self.encode(x)
	    x_reconst = self.decode(latent)
	    reconst_error = self.reconst_error(x, x_reconst)
	    return reconst_error, {'reconstruction_error': reconst_error}

	def backward(self,x) :
		pass


#AutoRec Encoder with Regularisation in for
class AutoRec_Regularisation(AutoencoderModel) :

	def __init__(self,k,d,regularisation_param=1) :
		super().__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=d, out_features=k, bias=True),
            nn.Sigmoid()
        )

		self.decoder = nn.Sequential(
			nn.Linear(in_features=d, out_features=k, bias=True),
			nn.Sigmoid()
		)

		self.reconst_error = nn.MSELoss()

	def encode(self, x) :
	self.encoder(x)

	def decode(self, x) :
		self.decoder(x)

	def regularisation(self) :
		V = self.encoder.layer[0].weight
		W = self.decoder.layer[0].weight 
		return torch.divide(regularisation_param,2) * (torch.square(torch.norm(W))+torch.square(torch.norm(V)))

    def forward(self, x):
	    latent = self.encode(x)
	    x_reconst = self.decode(latent)
	    reconst_error = self.reconst_error(x, x_reconst) + self.regularisation() 
	    return reconst_error, {'reconstruction_error': reconst_error}

	def backward(self, x) :
		pass









	
