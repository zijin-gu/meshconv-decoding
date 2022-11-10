import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from utils4image.dataloaders import NSDFeatureDataset
from utils4image.models import Brain2FeatureMeshPool
from utils4image.utils import load_generator, save_checkpoint
import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=int, default=1)
	parser.add_argument("--batchsize", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--epoch", type=int, default=100)
	parser.add_argument("--mapping", type=str, default='mesh')
	parser.add_argument("--optim", type=str, default='adamw')
	parser.add_argument("--dataset", type=str, default='imagenet')
	parser.add_argument("--indim", type=int, default=16, help='in dim for the first mesh conv')
	parser.add_argument("--fdim", type=int, default=16, help='hidden dim for mesh conv')
	parser.add_argument("--n_hidden_layer", type=int, default=3, help='number of hidden layers')
	parser.add_argument("--decay", type=float, default=0.01, help='weight decay')
	parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate')
	args = parser.parse_args()

	device = "cuda"
	subject = args.subject    
	lr = args.lr   
	n_epochs = args.epoch 
	optim = args.optim    
	mapping = args.mapping
	dataset = args.dataset
	decay = args.decay
	dropout_rate = args.dropout 
	fdim = args.fdim  
	indim = args.indim  
	n_hidden_layer = args.n_hidden_layer
	batch_size = args.batchsize
	ckpt_dir = f'./decoding_ckpt/S{subject}/feature_decoding/'
	samp_dir = f'./decoding_sample/S{subject}/feature_decoding/'
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(samp_dir):
		os.makedirs(samp_dir)

	model_base = '%s_%s_lr%s_dc%s_dp%s_fd%d_ind%d_layer%d'%(mapping, optim, "{:.0e}".format(lr),"{:.0e}".format(decay), "{:.0e}".format(dropout_rate), fdim, indim, n_hidden_layer)  

	train_dataset = NSDFeatureDataset(mode='train', test_subject=subject)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataset = NSDFeatureDataset(mode='val', test_subject=subject)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	
	model = Brain2FeatureMeshPool(fdim=fdim, indim=indim, n_hidden_layer=n_hidden_layer, dropout_rate=dropout_rate).to(device)
	proj_dir = '/home/zg243/image_generation/ic_gan/'
	generator = load_generator(f'icgan_biggan_{dataset}_res256', proj_dir+'pretrained_models', 'biggan', device)

	criterion = nn.MSELoss() 
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

	def train_step(data, target):
		model.train()
		data, target = data.to(device), target.to(device)  
		prediction = model(data)
		loss = criterion(prediction, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		return loss.item()

	def val_step():
		model.eval()
		pred, true, val_loss = [], [], 0
		for idx, (data, target) in enumerate(val_loader):
			data, target = data.to(device), target.to(device)  
			prediction = model(data)
			loss = criterion(prediction, target)

			val_loss += loss.item()
			pred.append(prediction.detach().cpu())
			true.append(target.detach().cpu())
			
		return torch.vstack(pred), torch.vstack(true), val_loss/(idx+1)

	results = {'train_loss':[], 'val_loss':[]}
	best_loss = 1e5
	for epoch in range(n_epochs):
		# training
		total_loss = 0

		for batch_idx, (data, target) in enumerate(train_loader):
			loss = train_step(data, target)
			total_loss += loss
    
			print("Training [{}:{}/{}] LOSS={:.2} <LOSS>={:.2}".format(
				epoch, batch_idx, len(train_loader), loss, total_loss / (batch_idx + 1)))

			if batch_idx % 100 == 0:
				results['train_loss'].append(total_loss / (batch_idx + 1))
				# val
				val_pred, val_true, val_loss = val_step()
				print("Validation [Epoch {} Test] <LOSS>={:.2}".format(epoch, val_loss))
				results['val_loss'].append(val_loss)
				np.save(ckpt_dir + model_base + '_results.npy', results)

				# generate samples
				zs = torch.empty(10, generator.dim_z,).normal_(mean=0, std=1.0).to(device)
				true_img = generator(zs, None, val_true[:10].to(device)).detach().cpu()
				pred_img = generator(zs, None, val_pred[:10].to(device)).detach().cpu()

				if val_loss <= best_loss:
					best_loss = val_loss  
					save_checkpoint(model, optimizer, epoch+1, model_base+'_best.pt', ckpt_dir)
					utils.save_image(torch.cat([true_img[:10], pred_img[:10]], 0),
									samp_dir+model_base+f"_val_best.png",nrow=10, 
									normalize=True, value_range=(-1, 1))
		
		save_checkpoint(model, optimizer, epoch, model_base+'_last.pt', ckpt_dir)
		utils.save_image(torch.cat([true_img[:10], pred_img[:10]], 0),
						samp_dir+model_base+f"_val_last.png",nrow=10, 
						normalize=True, value_range=(-1, 1))
