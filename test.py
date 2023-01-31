import torch
import numpy as np
import os
from utils4image.dataloaders import NSDImageDataset
from torch.utils.data import DataLoader
from utils4image.utils import load_generator
from utils4image.eva_utils import load_meshmodel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

subject = args.subject
batch_size = args.batch_size

scale = 1

device = torch.device('cuda')
mapping = 'meshpool'
optim = 'adamw'
lr = 1e-4
decay = 0.1
dropout_rate = 0.5
fdim = 32
indim = 32
n_hidden_layer = 3
recon_w = 1e-6
kld_w = 1e-8
annealing_epochs = 10
kld_start_epoch = 0
in_feature_w = 0
out_feature_w = 1
b2f_fix = True
variation = True
combined_type = 'variation' if variation else 'direct'
factor = 1

ckpt_dir = f'./decoding_ckpt/S{subject}/image_decoding/{combined_type}/'
model_base = '%s_%s_lr%s_dc%s_dp%s_fd%d_ind%d_layer%d_rw%s_ifw%s_ofw%s_kldw%s_ae%d_kldse%d_vsf%s'% \
(mapping, optim, "{:.0e}".format(lr),"{:.0e}".format(decay), "{:.0e}".format(dropout_rate),
fdim, indim, n_hidden_layer, "{:.0e}".format(recon_w),  "{:.0e}".format(in_feature_w), 
"{:.0e}".format(out_feature_w), "{:.0e}".format(kld_w), annealing_epochs, kld_start_epoch, "{:.0e}".format(factor))

model_base = (model_base + '_fixb2f') if b2f_fix else (model_base + '_ftb2f')
model = load_meshmodel(subject, 'image', fdim, indim, n_hidden_layer, dropout_rate, model_base).to(device)

test_dataset = NSDImageDataset(mode='test', test_subject=subject)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pred_feat = []
pred_mu = []
pred_logvar = []
for i, (data, target, target_f) in enumerate(test_loader):
    print(i)
    pred, feat, mu, logvar = model(data.to(device))
    pred_feat.append(feat.detach().cpu())
    pred_mu.append(mu.detach().cpu())
    pred_logvar.append(logvar.detach().cpu())

pred_mu = torch.vstack(pred_mu)
pred_logvar = torch.vstack(pred_logvar)*scale
pred_feat = torch.vstack(pred_feat)

pred_muvar = {'mu':pred_mu.numpy(), 'logvar':pred_logvar.numpy()}
n_pf_pz = [] # (10, 1000, 3, 256, 256)
for n in range(10):
    print(n)
    #torch.manual_seed(seed=n)
    pf_pz = []
    for i in range(len(test_loader)):
        print(i)
        pf = pred_feat[i*batch_size:(i+1)*batch_size].to(device)
        pz = model.brain2noise.reparametrize(pred_mu[i*batch_size:(i+1)*batch_size].to(device), pred_logvar[i*batch_size:(i+1)*batch_size].to(device))
        img = model.generator(pz, None, pf).detach().cpu().numpy()
        pf_pz.append(img)
    pf_pz = np.vstack(pf_pz)
    pf_pz = (pf_pz+1)/2
    n_pf_pz.append(pf_pz)

n_pf_pz = np.moveaxis(np.array(n_pf_pz), 2, -1)


# pf_pz = [] #(1000, 3, 256, 256)
# for i in range(len(test_loader)):
#     print(i)
#     pf = pred_feat[i*batch_size:(i+1)*batch_size].to(device)
#     pz = pred_mu[i*batch_size:(i+1)*batch_size].to(device)
#     img = model.generator(pz, None, pf).detach().cpu().numpy()
#     pf_pz.append(img)
# pf_pz = np.vstack(pf_pz)
# pf_pz = (pf_pz+1)/2

# pf_pz = np.moveaxis(pf_pz, 1, -1)

res_dir = f'./decoding_result/S{subject}/image_decoding/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

#np.save(res_dir + model_base + f'_pred_imgs_{scale}var.npy', n_pf_pz)
np.save(res_dir + model_base + f'_pred_imgs.npy', n_pf_pz)
