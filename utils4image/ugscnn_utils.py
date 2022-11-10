import torch
from torch import nn
import os 
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import pickle
from scipy import sparse
import numpy as np


class Down(nn.Module):
	def __init__(self, in_ch, out_ch, level, mesh_dir, vertices, bias=True):
		super().__init__()
		"""use mesh_file to perform convolution to the next coarser resolution mesh"""
		self.conv = ResPoolBlock(in_ch, in_ch, out_ch, level+1, True, mesh_dir, vertices)

	def forward(self, x):
		x = self.conv(x)
		return x
    
class Down_noReLU(nn.Module):
	def __init__(self, in_ch, out_ch, level, mesh_dir, vertices, bias=True):
		super().__init__()
		"""use mesh_file to perform convolution to the next coarser resolution mesh"""
		self.conv = ResPoolBlock_noReLU(in_ch, in_ch, out_ch, level+1, True, mesh_dir, vertices)

	def forward(self, x):
		x = self.conv(x)
		return x

class _MeshConv(nn.Module):
	def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
		assert stride in [1, 2]
		super(_MeshConv, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.ncoeff = 4
		self.coeffs = Parameter(torch.Tensor(out_channels, in_channels, self.ncoeff))
		self.set_coeffs()

		pkl = pickle.load(open(mesh_file, "rb"))
		self.pkl = pkl
		self.nv = self.pkl['V'].shape[0]
		G = sparse2tensor(pkl['G'])  # gradient matrix V->F, 3#F x #V
		NS = torch.tensor(pkl['NS'], dtype=torch.float32)  # north-south vector field, #F x 3
		EW = torch.tensor(pkl['EW'], dtype=torch.float32)  # east-west vector field, #F x 3
		self.register_buffer("G", G)
		self.register_buffer("NS", NS)
		self.register_buffer("EW", EW)
		
	def set_coeffs(self):
		n = self.in_channels * self.ncoeff
		stdv = 1. / math.sqrt(n)
		self.coeffs.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

class MeshConv(_MeshConv):
	def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
		super(MeshConv, self).__init__(in_channels, out_channels, mesh_file, stride, bias)
		pkl = self.pkl
		if stride == 2:
			self.nv_prev = pkl['nv_prev']
			L = sparse2tensor(pkl['L'].tocsr()[:self.nv_prev].tocoo()) # laplacian matrix V->V
			F2V = sparse2tensor(pkl['F2V'].tocsr()[:self.nv_prev].tocoo())  # F->V, #V x #F
		else: # stride == 1
			self.nv_prev = pkl['V'].shape[0]
			L = sparse2tensor(pkl['L'].tocoo())
			F2V = sparse2tensor(pkl['F2V'].tocoo())
		self.register_buffer("L", L)
		self.register_buffer("F2V", F2V)
		
	def forward(self, input):
		# compute gradient
		grad_face = spmatmul(input, self.G)
		grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2) # gradient, 3 component per face
		laplacian = spmatmul(input, self.L)
		identity = input[..., :self.nv_prev]
		grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
		grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)
		grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
		grad_vert_ns = spmatmul(grad_face_ns, self.F2V)

		feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]

		out = torch.stack(feat, dim=-1)
		out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
		out += self.bias.unsqueeze(-1)
		return out

class ResPoolBlock_noReLU(nn.Module):
	def __init__(self, in_chan, neck_chan, out_chan, level, coarsen, mesh_dir, vertices):
		super().__init__()
		l = level-1 if coarsen else level
		self.coarsen = coarsen
		mesh_file = os.path.join(mesh_dir, "icosphere_{}.pkl".format(vertices[l]))
		self.conv1 = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
		self.conv2 = MeshConv(neck_chan, neck_chan, mesh_file=mesh_file, stride=1)
		self.conv3 = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=True)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		self.bn1 = nn.BatchNorm1d(neck_chan)
		self.bn2 = nn.BatchNorm1d(neck_chan)
		self.bn3 = nn.BatchNorm1d(out_chan)
		self.nv_prev = self.conv2.nv_prev
		self.pool = MaxPool(mesh_dir, level, vertices)
		self.diff_chan = (in_chan != out_chan)

		if coarsen:
			self.seq1 = nn.Sequential(self.conv1, self.pool, self.bn1,
									  self.conv2, self.bn2, 
									  self.conv3, self.bn3)
		else:
			self.seq1 = nn.Sequential(self.conv1, self.bn1, 
									  self.conv2, self.bn2,
									  self.conv3, self.bn3)

		if self.diff_chan or coarsen:
			self.conv_ = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
			self.bn_ = nn.BatchNorm1d(out_chan)
			if coarsen:
				self.seq2 = nn.Sequential(self.conv_, self.pool, self.bn_)
			else:
				self.seq2 = nn.Sequential(self.conv_, self.bn_)

	def forward(self, x):
		if self.diff_chan or self.coarsen:
			x2 = self.seq2(x)
		else:
			x2 = x
		x1 = self.seq1(x)
		out = x1 + x2
		#out = self.leaky_relu(out)
		return out

    
class ResPoolBlock(nn.Module):
	def __init__(self, in_chan, neck_chan, out_chan, level, coarsen, mesh_dir, vertices):
		super().__init__()
		l = level-1 if coarsen else level
		self.coarsen = coarsen
		mesh_file = os.path.join(mesh_dir, "icosphere_{}.pkl".format(vertices[l]))
		self.conv1 = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
		self.conv2 = MeshConv(neck_chan, neck_chan, mesh_file=mesh_file, stride=1)
		self.conv3 = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=True)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		self.bn1 = nn.BatchNorm1d(neck_chan)
		self.bn2 = nn.BatchNorm1d(neck_chan)
		self.bn3 = nn.BatchNorm1d(out_chan)
		self.nv_prev = self.conv2.nv_prev
		self.pool = MaxPool(mesh_dir, level, vertices)
		self.diff_chan = (in_chan != out_chan)

		if coarsen:
			self.seq1 = nn.Sequential(self.conv1, self.pool, self.bn1, self.relu,
									  self.conv2, self.bn2, self.relu,
									  self.conv3, self.bn3)
		else:
			self.seq1 = nn.Sequential(self.conv1, self.bn1, self.relu,
									  self.conv2, self.bn2, self.relu,
									  self.conv3, self.bn3)

		if self.diff_chan or coarsen:
			self.conv_ = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
			self.bn_ = nn.BatchNorm1d(out_chan)
			if coarsen:
				self.seq2 = nn.Sequential(self.conv_, self.pool, self.bn_)
			else:
				self.seq2 = nn.Sequential(self.conv_, self.bn_)

	def forward(self, x):
		if self.diff_chan or self.coarsen:
			x2 = self.seq2(x)
		else:
			x2 = x
		x1 = self.seq1(x)
		out = x1 + x2
		out = self.relu(out)
		return out


class MaxPool(nn.Module):
	def __init__(self, mesh_dir, level, vertices):
		super().__init__()
		self.level = level

		if self.level > 0:
			vertices_to_prev_lvl_file = os.path.join(mesh_dir, "ico%s_to_ico%s_vertices.npy" % (vertices[level-1], vertices[level]))
			self.vertices_to_prev_lvl = np.load(vertices_to_prev_lvl_file)

			neihboring_patches_file = os.path.join(mesh_dir, "ico%s_neighbor_patches.npy" % (vertices[level-1]))
			self.neihboring_patches = np.load(neihboring_patches_file)

	def forward(self, x):
		tmp = x[..., self.vertices_to_prev_lvl]
		out, indices = torch.max(tmp[:, :, self.neihboring_patches], -1)
		return out

"""
from https://github.com/maxjiang93/ugscnn
"""

def sparse2tensor(m):
	"""
	Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
	"""
	assert(isinstance(m, sparse.coo.coo_matrix))
	i = torch.LongTensor([m.row, m.col])
	v = torch.FloatTensor(m.data)
	return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))

def spmatmul(den, sp):
	"""
	den: Dense tensor of shape batch_size x in_chan x #V
	sp : Sparse tensor of shape newlen x #V
	"""
	batch_size, in_chan, nv = list(den.size())
	new_len = sp.size()[0]
	den = den.permute(2, 1, 0).contiguous().view(nv, -1)
	res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
	return res
