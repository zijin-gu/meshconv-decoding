from .models import Brain2FeatureMeshPool, Brain2Image, Brain2NoiseVarMeshPool
import os
import torch
from torch.nn.parameter import Parameter

def load_meshmodel(subject, element, fdim, indim, n_hidden_layer, dropout_rate, restore_file, variation=True, mtype='best', device=torch.device('cpu')):
	prefix = f'./decoding_ckpt/S{subject}/{element}_decoding/'
	if element == 'feature':
		model = Brain2FeatureMeshPool(fdim=fdim, indim=indim, n_hidden_layer=n_hidden_layer, dropout_rate=dropout_rate)
	elif element == 'image':
		b2f = Brain2FeatureMeshPool(fdim=fdim, indim=indim, n_hidden_layer=n_hidden_layer, dropout_rate=dropout_rate)
		if variation:
			b2z = Brain2NoiseVarMeshPool(indim=indim, fdim=fdim, n_hidden_layer=n_hidden_layer, factor=1)
		model = Brain2Image(b2f, b2z,  b2f_fix=True, generator_fix=True, variation=variation)
		prefix = prefix + 'variation/' if variation else prefix
	restore_path = os.path.join(prefix, restore_file + f'_{mtype}.pt')
	resume_dict = torch.load(restore_path, map_location=torch.device('cpu'))
	state_dict = resume_dict['state_dict']
	print(resume_dict['epoch'])
	model_state = model.state_dict()
	for name, param in state_dict.items():
		if name not in model_state:
			continue
		if 'none' in name:
			continue
		if isinstance(param, Parameter):
			# backwards compatibility for serialized parameters
			param = param.data
		model_state[name].copy_(param)
	model.eval()
	return model.to(device)
