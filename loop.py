
import time
import pdb
import copy
import yaml
import argparse

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

from mamlpytorch.tasks.sinusoid_tasks import SinusoidTaskDistribution
from mamlpytorch.networks import SinusoidModel

'''
Theta: Meta-parameters
Theta': Task-parameters
'''

def replace_grad(param_grad, param_name):
	def replace_grad_(module):
		return param_grad[param_name]

	return replace_grad_


def main(cfg):

	np.random.seed(1)
	torch.manual_seed(1)

	writer = SummaryWriter(log_dir = 'runs/hooks2')

	dataset = cfg.get('dataset', 'sinusoid')

	if dataset == "sinusoid":
		
		model = SinusoidModel()
		task_distribution = SinusoidTaskDistribution()
		loss_function = nn.MSELoss()

	inner_lr = cfg['inner']['lr'] # 1e-3
	meta_lr = cfg['meta']['lr'] # 1e-2

	print(cfg['meta']['lr'])
	meta_optimizer = optim.Adam(model.parameters(), lr = cfg['meta']['lr'])

	num_tasks = cfg['meta']['batch_size'] # 25
	num_datapoints = cfg['inner']['batch_size'] # 10

	num_meta_iterations = cfg['meta']['training_iterations'] # 50000
	num_inner_training_iterations = cfg['inner']['training_iterations'] # 1

	start = time.time()
	
	for meta_iter in range(cfg['meta']['training_iterations']):
	
		'''
		Step 3
		'''
		tasks = task_distribution.sample_batch(batch_size = cfg['meta']['batch_size'])
		
		meta_loss = 0
		task_gradients = []

		for i, task in enumerate(tasks):
		
			'''
			Step 5
			'''
			x_support, y_support = task.sample_batch(batch_size = cfg['inner']['batch_size'])
			
			'''
			Step 6
			'''
			fast_weights = OrderedDict(model.named_parameters())
			y_support_prime = model.functional_forward(x_support, fast_weights)
			inner_train_loss = loss_function(y_support_prime, y_support)
			inner_gradient = torch.autograd.grad(inner_train_loss, fast_weights.values(), create_graph = False)

			'''
			Step 7
			'''
			fast_weights = OrderedDict(
				(name, param - inner_lr * grad)
				for ((name, param), grad) in zip(fast_weights.items(), inner_gradient))

			'''
			Step 8
			'''
			x_query, y_query = task.sample_batch(batch_size = cfg['inner']['batch_size'])
			
			'''
			Save gradients for each task
			'''
			y_query_prime = model.functional_forward(x_query, fast_weights)
			inner_test_loss = loss_function(y_query_prime, y_query)
			inner_test_gradient = torch.autograd.grad(inner_test_loss, fast_weights.values())
			named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), inner_test_gradient)}

			task_gradients.append(named_grads) 

			meta_loss += inner_test_loss
		
		if (meta_iter + 1) % 50 == 0:
			writer.add_scalar('Loss/QuerySet', meta_loss.item() / cfg['meta']['batch_size'], 
				meta_iter)
	
		'''
		Step 10
		'''
		meta_gradient = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim = 0) 
								for k in task_gradients[0].keys()}
		
		hooks = []
		for name, param in model.named_parameters():
			hooks.append(
				param.register_hook(replace_grad(meta_gradient, name)))

		meta_optimizer.zero_grad()
		dummy_pred = model(torch.zeros(1, 1))
		dummy_loss = loss_function(dummy_pred, torch.zeros(1, 1))
		'''
		Replacing gradient of every parameter in the meta-model using a backward hook
		'''
		dummy_loss.backward()
		meta_optimizer.step()

		for h in hooks:
			h.remove()

	print(time.time() - start)		

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description = "config")
	parser.add_argument('--config', 
					nargs = '?', 
					type = str, 
					default = 'config.yml')

	args = parser.parse_args()

	with open(args.config) as f_in:
		cfg = yaml.safe_load(f_in)

	main(cfg)