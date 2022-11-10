import sys
sys.path.append("/home/zg243/image_generation/ic_gan/")
import os 
import BigGAN_PyTorch.utils as biggan_utils
import inference.utils as inference_utils
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn

def load_generator(exp_name, root_path, backbone, device="cpu"):
	parser = biggan_utils.prepare_parser()
	parser = biggan_utils.add_sample_parser(parser)
	parser = inference_utils.add_backbone_parser(parser)

	args = ["--experiment_name", exp_name]
	args += ["--base_root", root_path]
	args += ["--model_backbone", backbone]

	config = vars(parser.parse_args(args=args))

	# Load model and overwrite configuration parameters if stored in the model
	config = biggan_utils.update_config_roots(config, change_weight_folder=False)
	generator, config = inference_utils.load_model_inference(config, device=device)
	biggan_utils.count_parameters(generator)
	generator.eval()

	return generator


def save_checkpoint(model, optimizer, epoch, fname, output_dir):
    state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() != "torch.cuda.sparse.FloatTensor"]
    state_dict_no_sparse = OrderedDict(state_dict_no_sparse)

    checkpoint = {
        'epoch': epoch,
        'state_dict': state_dict_no_sparse,
        #'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(output_dir, fname))


def featurespace_loss(true_feat, pred, extractor):
	"""
	Args:
		true_feat: the true image feature (no need to extract from image again)
		pred: the decoded image (need to extract feature first)
		extractor: swav layer extractor
	Returns: feature space loss
	"""
	pred_feat = extractor(pred).view(pred.shape[0], -1)  # unnormalized
	pred_feat = F.normalize(pred_feat, dim=1, p=2)  # normalized true feats
	return F.mse_loss(true_feat, pred_feat)


def load_swav():
	swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')
	return nn.Sequential(*list(swav.children())[:-1]).eval()
