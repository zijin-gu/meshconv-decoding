import torch
from torch import nn


class Brain2Image(nn.Module):
	def __init__(self, device):
		super().__init__()
		self.brain2inputs = Brain2Inputs()
		self.generator = get_model('icgan_biggan_imagenet_res256', './pretrained_models', 'biggan', device=device)

	def forward(self, x):
		feat, z = self.brain2inputs(x)
		y = self.generator(z, None, feat)

    
    
import torch



# load feature extractor SWAV
swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')
layer4_extractor = nn.Sequential(*list(swav.children())[:-1]).to(device)
layer4_extractor.eval()

criterion = nn.MSELoss()

for i, (act, img) in enumerate(loader):
	model.zero_grad()
	act, img = act.to(device), img.to(device)
	gen_img = model(act)
	gen_feats = layer4_extractor(gen_img).flatten()
	true_feats = layer4_extractor(img).flatten()

	loss = criterion(gen_feats, orig_feats)
	loss.backward()
	optimizer.step()
	
