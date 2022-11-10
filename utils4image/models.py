import torch
from torch import nn
import os 
#from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .utils import load_generator, featurespace_loss, load_swav
from .ugscnn_utils import *
#import math
#import pickle
#from scipy import sparse
#import numpy as np
from torch.autograd import Variable

class Brain2FeatureMeshPool(nn.Module):
	def __init__(self, mesh_dir='/home/zg243/SharedRep/mesh_assets/meshes/100-500-2k-8k-32k/', 
				in_ch=2, indim=1, fdim=16, max_level=4, embed_dim=2048, n_hidden_layer=3,
				vertices=['100','500','2k','8k', '32k'], dropout_rate=0.5):
		super().__init__()
		self.mesh_dir = mesh_dir 
		self.vertices = vertices    
		self.fdim = fdim
		self.embed_dim = embed_dim
		self.n_hidden_layer = n_hidden_layer
		self.levels = max_level
		self.in_conv = MeshConv(in_ch, indim, self.__meshfile(max_level), stride=1)
		self.in_bn = nn.BatchNorm1d(indim)
		self.relu = nn.ReLU(inplace=True)
		self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
		self.block1 = Down(indim, fdim*4, max_level-1, mesh_dir, vertices)
		self.block2 = Down(fdim*4, fdim*16, max_level-2, mesh_dir, vertices)
		self.block3 = Down(fdim*16, fdim*64, max_level-3, mesh_dir, vertices)
		self.block4 = Down(fdim*64, fdim*128, max_level-4, mesh_dir, vertices)
		if n_hidden_layer == 3:
			self.blocks = [self.block1, self.block2, self.block3]
		else:
			self.blocks = [self.block1, self.block2, self.block3, self.block4]
		self.avg = nn.AvgPool1d(kernel_size=self.blocks[-1].conv.nv_prev) # output shape batch x channels x 1
		self.out_layer = nn.Linear((2**(n_hidden_layer+3))*fdim, embed_dim)
		self.out_bn = nn.BatchNorm1d(embed_dim)  
		self.out_block = nn.Sequential(self.out_layer, self.out_bn, self.relu) 
		self.dropout_rate = dropout_rate  
        
	def forward(self, x):
		x = self.in_block(x)
		for block in self.blocks:
			x = block(x)
		x = torch.squeeze(self.avg(x), dim=-1)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.out_block(x)
		x = F.normalize(x, dim=1, p=2)
		return x
    
	def __meshfile(self, i):
		return os.path.join(self.mesh_dir, "icosphere_{}.pkl".format(self.vertices[i]))
    

class Brain2NoiseMeshPool(nn.Module):
	def __init__(self, mesh_dir='/home/zg243/SharedRep/mesh_assets/meshes/100-500-2k-8k-32k/', 
				 in_ch=2, indim=2, fdim=16, n_hidden_layer=3, max_level=4,
				 vertices=['100','500','2k','8k', '32k'], dropout_rate=0.5, latent_dim=119):
		super().__init__()
		self.mesh_dir = mesh_dir 
		self.vertices = vertices    
		self.fdim = fdim
		self.indim = indim
		self.n_hidden_layer = n_hidden_layer
		self.levels = max_level
		self.in_conv = MeshConv(in_ch, indim, self.__meshfile(max_level), stride=1)
		self.in_bn = nn.BatchNorm1d(indim)
		self.relu = nn.ReLU(inplace=True)
		self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
		self.block1 = Down(indim, fdim * 4, max_level - 1, mesh_dir, vertices)
		self.block2 = Down(fdim * 4, fdim * 16, max_level - 2, mesh_dir, vertices)
		self.block3 = Down(fdim * 16, fdim * 64, max_level - 3, mesh_dir, vertices)
		self.block4 = Down(fdim * 64, fdim * 128, max_level - 4, mesh_dir, vertices)
		if n_hidden_layer == 3:
			self.blocks = [self.block1, self.block2, self.block3]
		else:
			self.blocks = [self.block1, self.block2, self.block3, self.block4]
		self.avg = nn.AvgPool1d(kernel_size=self.block3.conv.nv_prev) # output shape batch x channels x 1
		self.dropout_rate = dropout_rate

		self.out_layer = nn.Linear((2**(n_hidden_layer+3)) * fdim, latent_dim)
		self.z_mag = np.sqrt(2)*1.386831e80/1.801679e79 # \sqrt(2)*Gamma(60)/Gamma(59.5)

	def forward(self, x):
		x = self.in_block(x)
		for block in self.blocks:
			x = block(x)
		x = torch.squeeze(self.avg(x), dim=-1)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		z = self.out_layer(x)
		z = self.z_mag*F.normalize(z, p=2)
		return z

	def __meshfile(self, i):
		return os.path.join(self.mesh_dir, "icosphere_{}.pkl".format(self.vertices[i]))

class Brain2NoiseVarMeshPool(nn.Module):
	def __init__(self, mesh_dir='/home/zg243/SharedRep/mesh_assets/meshes/100-500-2k-8k-32k/', 
				 in_ch=2, indim=2, fdim=16, n_hidden_layer=3, max_level=4,
				 vertices=['100','500','2k','8k', '32k'], dropout_rate=0.5, latent_dim=119, factor=1):
		super().__init__()
		self.mesh_dir = mesh_dir 
		self.vertices = vertices    
		self.fdim = fdim
		self.indim = indim
		self.n_hidden_layer = n_hidden_layer
		self.levels = max_level
		self.in_conv = MeshConv(in_ch, indim, self.__meshfile(max_level), stride=1)
		self.in_bn = nn.BatchNorm1d(indim)
		self.relu = nn.ReLU(inplace=True)
		self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
		self.block1 = Down(indim, fdim * 4, max_level - 1, mesh_dir, vertices)
		self.block2 = Down(fdim * 4, fdim * 16, max_level - 2, mesh_dir, vertices)
		self.block3 = Down(fdim * 16, fdim * 64, max_level - 3, mesh_dir, vertices)
		self.block4 = Down(fdim * 64, fdim * 128, max_level - 4, mesh_dir, vertices)
		if n_hidden_layer == 3:
			self.blocks = [self.block1, self.block2, self.block3]
		else:
			self.blocks = [self.block1, self.block2, self.block3, self.block4]
		self.avg = nn.AvgPool1d(kernel_size=self.block3.conv.nv_prev) # output shape batch x channels x 1
		self.dropout_rate = dropout_rate

		self.fc1 = nn.Linear((2**(n_hidden_layer+3)) * fdim, latent_dim)
		self.fc2 = nn.Linear((2**(n_hidden_layer+3)) * fdim, latent_dim)
		self.chisq_mean = 10.886 # \sqrt(2)*Gamma(60)/Gamma(59.5)
		self.radius = 12.9 # R: sqrt( qchisq( pchisq( 9,df=1 ), df = 119))
		self.factor = factor

	def encode(self, x):
		x = self.in_block(x)
		for block in self.blocks:
			x = block(x)
		x = torch.squeeze(self.avg(x), dim=-1)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		mu = self.fc1(x)
		#mu = self.chisq_mean*F.normalize(mu, p=2) # scale mean to be on the hypersphere
		logvar = self.fc2(x)
		logvar -= torch.log(torch.tensor(self.factor)**2)
		return mu, logvar

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		if torch.cuda.is_available():
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()            
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparametrize(mu, logvar)
		return z, mu, logvar

	def __meshfile(self, i):
		return os.path.join(self.mesh_dir, "icosphere_{}.pkl".format(self.vertices[i]))


class Brain2Image(nn.Module):
	def __init__(self, b2f, b2z, dataset='imagenet', b2f_fix=True, generator_fix=True, variation=False):
		super().__init__()
		self.brain2feature = b2f
		self.brain2noise = b2z
		self.generator = load_generator(f'icgan_biggan_{dataset}_res256', '/home/zg243/image_generation/ic_gan/pretrained_models', 'biggan')
		self.feature_extractor = load_swav()
		self.variation = variation        
		if b2f_fix:        
			self.fix_weights(self.brain2feature)
		if generator_fix:        
			self.fix_weights(self.generator)
            
	def forward(self, x):
		feature = self.brain2feature(x)
        
		if self.variation:        
			z, mu, logvar = self.brain2noise(x)
			y_hat = self.generator(z, None, feature)
			return y_hat, feature, mu, logvar
		else:
			z = self.brain2noise(x)
			y_hat = self.generator(z, None, feature)
			return y_hat, feature, 0, 0

	def fix_weights(self, block):
		for param in block.parameters():
			param.requires_grad = False

	def compute_loss(self, y, y_hat, in_feature, in_feature_hat, mu=None, logvar=None, recon_w=1, in_feature_w=1, out_feature_w=1, kld_w=0):
		recon_loss = F.mse_loss(y, y_hat)
		in_feature_loss = F.mse_loss(in_feature, in_feature_hat)
		out_feature_loss = featurespace_loss(in_feature, y_hat, self.feature_extractor)
		if self.variation:  
			kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
		else:
			kld_loss = torch.zeros(recon_loss.size())
		loss = recon_w*recon_loss + in_feature_w*in_feature_loss + out_feature_w*out_feature_loss + kld_w*kld_loss
		return loss, (recon_loss, in_feature_loss, out_feature_loss, kld_loss)
  
