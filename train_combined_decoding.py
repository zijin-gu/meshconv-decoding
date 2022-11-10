import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from utils4image.dataloaders import NSDImageDataset
from utils4image.models import Brain2FeatureMeshPool, Brain2NoiseVarMeshPool, Brain2NoiseMeshPool, Brain2Image
from utils4image.utils import load_generator, save_checkpoint
from utils4image.eva_utils import load_meshmodel
import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=int, default=1)
	parser.add_argument("--batchsize", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--epoch", type=int, default=100)
	parser.add_argument("--mapping", type=str, default='meshpool')
	parser.add_argument("--optim", type=str, default='adamw')
	parser.add_argument("--dataset", type=str, default='imagenet')
	parser.add_argument("--indim", type=int, default=32, help='in dim for the first mesh conv')
	parser.add_argument("--fdim", type=int, default=32, help='hidden dim for mesh conv')
	parser.add_argument("--n_hidden_layer", type=int, default=3, help='number of hidden layers')
	parser.add_argument("--recon_w", type=float, default=1, help='weight for the reconstruction loss')
	parser.add_argument("--in_feature_w", type=float, default=1, help='weight for the in feature space loss')
	parser.add_argument("--out_feature_w", type=float, default=1, help='weight for the out feature space loss')
	parser.add_argument("--kld_w", type=float, default=1, help='weight for the KL divergence loss')
	parser.add_argument("--decay", type=float, default=0.1, help='weight decay')
	parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate')
	parser.add_argument("--annealing_epochs", type=float, default=20, help='annealing epochs')
	parser.add_argument("--kld_start_epoch", type=float, default=20, help='when to start kld loss')
	parser.add_argument("--b2f_fix", default=True, action='store_false', help='fix the pretrained feature decoder or not')
	parser.add_argument("--gen_fix", default=True, action='store_false', help='fix the pretrained generator or not')
	parser.add_argument("--variation", default=False, action='store_true', help='variational or not')
	parser.add_argument("--factor", default=1, type=float, help='scaling factor for the variance if using variational approach')
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
	recon_w = args.recon_w
	in_feature_w = args.in_feature_w
	out_feature_w = args.out_feature_w
	kld_w = args.kld_w
	batch_size = args.batchsize
	annealing_epochs = args.annealing_epochs
	kld_start_epoch = args.kld_start_epoch
	variation = args.variation
	combined_type = 'variation' if variation else 'direct'
	factor = args.factor
	b2f_fix = args.b2f_fix
	gen_fix = args.gen_fix
	ckpt_dir = f'./decoding_ckpt/S{subject}/image_decoding/{combined_type}/'
	samp_dir = f'./decoding_sample/S{subject}/image_decoding/{combined_type}/'
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(samp_dir):
		os.makedirs(samp_dir)

	model_base = '%s_%s_lr%s_dc%s_dp%s_fd%d_ind%d_layer%d_rw%s_ifw%s_ofw%s_kldw%s_ae%d_kldse%d_vsf%s'%(mapping, optim, "{:.0e}".format(lr),
																			   "{:.0e}".format(decay), "{:.0e}".format(dropout_rate),
																			   fdim, indim, n_hidden_layer,
																			   "{:.0e}".format(recon_w),  "{:.0e}".format(in_feature_w), "{:.0e}".format(out_feature_w),
																			   "{:.0e}".format(kld_w), annealing_epochs, kld_start_epoch, "{:.0e}".format(factor))
	model_base = (model_base + '_fixb2f') if b2f_fix else (model_base + '_ftb2f')
	train_dataset = NSDImageDataset(mode='train', test_subject=subject)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataset = NSDImageDataset(mode='val', test_subject=subject)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	
	feat_fdim, feat_indim, feat_n_hidden_layer = 32, 32, 3
	restore_file = '%s_%s_lr%s_dc%s_dp%s_fd%d_ind%d_layer%d'%(mapping, optim, "{:.0e}".format(1e-3),
															"{:.0e}".format(decay), "{:.0e}".format(dropout_rate), 
															feat_fdim, feat_indim, feat_n_hidden_layer) 
	b2f = load_meshmodel(subject, 'feature', feat_fdim, feat_indim, feat_n_hidden_layer, dropout_rate, restore_file) 

	if variation:
		b2z = Brain2NoiseVarMeshPool(indim=indim, fdim=fdim, n_hidden_layer=n_hidden_layer, factor=factor)
	else:
		b2z = Brain2NoiseMeshPool(indim=indim, fdim=fdim, n_hidden_layer=n_hidden_layer)
	model = Brain2Image(b2f, b2z,  b2f_fix=b2f_fix, generator_fix=gen_fix, variation=variation).to(device)
	print('Variation approach is ', variation)
	print('Feature decoder is fixed: ', b2f_fix, ' Generator is fixed: ', gen_fix)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

	def train_step(data, target, target_f, annealed_kld_w):
		model.train()
		data, target, target_f = data.to(device), target.to(device), target_f.to(device)
		prediction, feature, mu, logvar = model(data)
		loss, (recon_loss, in_feat_loss, out_feature_loss, kld_loss) = model.compute_loss(target, prediction, target_f, feature, mu, logvar, recon_w, in_feature_w, out_feature_w, annealed_kld_w)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		return loss.item(), recon_loss.item(), in_feat_loss.item(), out_feature_loss.item(), kld_loss.item()

	def val_step(annealed_kld_w):
		model.eval()
		pred, true, val_loss, val_recon_loss, val_infeat_loss, val_outfeat_loss, val_kld_loss = [], [], 0, 0, 0, 0, 0
		for idx, (data, target, target_f) in enumerate(val_loader):
			data, target, target_f = data.to(device), target.to(device), target_f.to(device)
			prediction, feature, mu, logvar = model(data)
			loss, (recon_loss, infeat_loss, outfeat_loss, kld_loss) = model.compute_loss(target, prediction, target_f, feature, mu, logvar, recon_w, in_feature_w, out_feature_w, annealed_kld_w)

			val_loss += loss.item()
			val_recon_loss += recon_loss.item()
			val_infeat_loss += infeat_loss.item()
			val_outfeat_loss += outfeat_loss.item()
			val_kld_loss += kld_loss.item()
			pred.append(prediction.detach().cpu())
			true.append(target.detach().cpu())
			
		return torch.vstack(pred), torch.vstack(true), val_loss/(idx+1), val_recon_loss/(idx+1), val_infeat_loss/(idx+1), val_outfeat_loss/(idx+1), val_kld_loss/(idx+1)

	results = {'train_loss':[], 'train_recon_loss':[], 'train_infeat_loss':[], 'train_outfeat_loss':[], 'train_kld_loss':[],
			   'val_loss':[], 'val_recon_loss':[], 'val_infeat_loss':[], 'val_outfeat_loss':[], 'val_kld_loss':[]}
	best_loss = 1e5
	N_mini_batches = len(train_loader)
	for epoch in range(1, n_epochs+1):
		# training
		total_loss, total_recon_loss, total_infeat_loss, total_outfeat_loss, total_kld_loss = 0, 0, 0, 0, 0

		for batch_idx, (data, target, target_f) in enumerate(train_loader):                
			if (kld_w == 0) or (epoch <= kld_start_epoch):
				annealing_factor = 0
			else:
				annealing_factor = kld_w * (float(batch_idx + (epoch - kld_start_epoch - 1) * N_mini_batches + 1) /
												 float(annealing_epochs * N_mini_batches))
			loss, recon_loss, infeat_loss, outfeat_loss, kld_loss = train_step(data, target, target_f, annealing_factor)
			total_loss += loss
			total_recon_loss += recon_loss
			total_infeat_loss += infeat_loss
			total_outfeat_loss += outfeat_loss
			total_kld_loss += kld_loss

			print("Training [{}:{}/{}] LOSS={:.2} RECON={:.2} INFEAT={:.2} OUTFEAT={:.2} KLD={:.2} <LOSS>={:.2} ".format(
				epoch, batch_idx, len(train_loader), loss, recon_loss, infeat_loss, outfeat_loss, kld_loss, total_loss / (batch_idx + 1)))

			if batch_idx % 100 == 0:
				results['train_loss'].append(total_loss / (batch_idx + 1))
				results['train_recon_loss'].append(total_recon_loss / (batch_idx + 1))
				results['train_infeat_loss'].append(total_infeat_loss/ (batch_idx + 1))
				results['train_outfeat_loss'].append(total_outfeat_loss/ (batch_idx + 1))
				results['train_kld_loss'].append(total_kld_loss / (batch_idx + 1))

				# val
				val_pred, val_true, val_loss, val_recon_loss, val_infeat_loss, val_outfeat_loss, val_kld_loss = val_step(annealing_factor)
				print("Validation [Epoch {} Test] <LOSS>={:.2} <RECON>={:.2} <INFEAT>={:.2} <OUTFEAT>={:.2} <KLD>={:.2}".format(epoch, val_loss, val_recon_loss, val_infeat_loss, val_outfeat_loss, val_kld_loss))
				results['val_loss'].append(val_loss)
				results['val_recon_loss'].append(val_recon_loss)
				results['val_infeat_loss'].append(val_infeat_loss)
				results['val_outfeat_loss'].append(val_outfeat_loss)
				results['val_kld_loss'].append(val_kld_loss)
				np.save(ckpt_dir + model_base + '_results.npy', results)

				if val_loss <= best_loss:
					best_loss = val_loss  
					save_checkpoint(model, optimizer, epoch, model_base+'_best.pt', ckpt_dir)
					utils.save_image(torch.cat([val_true[:10], val_pred[:10]], 0),
									samp_dir+model_base+f"_val_best.png",nrow=10, 
									normalize=True, value_range=(-1, 1))
		
		save_checkpoint(model, optimizer, epoch, model_base+'_last.pt', ckpt_dir)
		utils.save_image(torch.cat([val_true[:10], val_pred[:10]], 0),
						samp_dir+model_base+f"_val_last.png",nrow=10, 
						normalize=True, value_range=(-1, 1))
