mapping = 'meshpool'
optim = 'adamw'
lr = 1e-4
decay = 0.1
dropout_rate = 0.5
fdim = 32
indim = 32
n_hidden_layer = 3
recon_w = 1e-7
kld_w = 1e-8
annealing_epochs = 10
kld_start_epoch = 0
in_feature_w = 0
out_feature_w = 1
b2f_fix = True
variation = True
combined_type = 'variation' if variation else 'direct'
factor = 1

feature_model_base = '%s_%s_lr%s_dc%s_dp%s_fd%d_ind%d_layer%d'%(mapping, optim, 
                                                                "{:.0e}".format(1e-3),"{:.0e}".format(decay), 
                                                                "{:.0e}".format(dropout_rate),fdim, indim, n_hidden_layer) 

combined_model_base = '%s_%s_lr%s_dc%s_dp%s_fd%d_ind%d_layer%d_rw%s_ifw%s_ofw%s_kldw%s_ae%d_kldse%d_vsf%s_fixb2f'% \
(mapping, optim, "{:.0e}".format(lr),"{:.0e}".format(decay), "{:.0e}".format(dropout_rate),
fdim, indim, n_hidden_layer, "{:.0e}".format(recon_w),  "{:.0e}".format(in_feature_w), 
"{:.0e}".format(out_feature_w), "{:.0e}".format(kld_w), annealing_epochs, kld_start_epoch, "{:.0e}".format(factor))
