
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

def replace_grad(param_grad, param_name):
	def replace_grad_(module):
		return param_grad[param_name]

	return replace_grad_


def main(cfg):

	np.random.seed(1)
	torch.manual_seed(1)

	writer = SummaryWriter()

	dataset = cfg.get('dataset', 'sinusoid')

	if dataset == "sinusoid":
		
		model = SinusoidModel()
		task_distribution = SinusoidTaskDistribution()
		loss_function = nn.MSELoss()

	meta_optimizer = optim.Adam(model.parameters(), lr = cfg['meta']['lr'])

	# start = time.time()
	
	for meta_iter in range(cfg['meta']['training_iterations']):
		
		'''
		Step 3
		'''
		tasks = task_distribution.sample_batch(batch_size = cfg['meta']['batch_size'])
		
		meta_loss = 0.
		meta_gradients = []

		for i, task in enumerate(tasks):
			
			'''
			Step 5
			'''
			x_support, y_support = task.sample_batch(batch_size = cfg['inner']['batch_size'])
			
			'''
			Step 6
			'''
			task_adapted_weights = OrderedDict(model.named_parameters())
			y_support_pred = model.functional_forward(x_support, fast_weights)
			task_support_loss = loss_function(y_support_pred, y_support)
			task_support_gradient = torch.autograd.grad(task_support_loss, task_adapted_weights.values())

			'''
			Step 7
			'''
			task_adapted_weights = OrderedDict(
				(name, param - inner_lr * grad)
				for ((name, param), grad) in zip(task_adapted_weights.items(), task_support_gradient))

			'''
			Step 8
			'''
			x_query, y_query = task.sample_batch(batch_size = cfg['inner']['batch_size'])
			
			'''
			Save gradients for each task
			'''
			y_query_pred = model.functional_forward(x_query, task_adapted_weights)
			task_query_loss = loss_function(y_query_pred, y_query)
			task_query_gradient = torch.autograd.grad(task_query_loss, task_adapted_weights.values(), create_graph = False)
			
			'''
			Naming the gradients
			'''
			task_query_gradient = {name: g for ((name, _), g) in zip(task_adapted_weights.items(), task_query_gradient)}
			
			meta_gradients.append(task_query_gradient) 

			meta_loss += task_query_loss
		
		if (meta_iter + 1) % cfg['logs']['writer_interval'] == 0:
			writer.add_scalar('Loss/QuerySet', meta_loss.item() / cfg['meta']['batch_size'], 
				meta_iter)
		
		'''
		Define a run ID and append to model path
		'''
		if (meta_iter + 1) % cfg['logs']['save_interval'] == 0:
			torch.save(model.state_dict(), cfg['logs']['model_save_path'])

		'''
		Step 10
		'''
		meta_gradient = {k: torch.stack([grad[k] for grad in meta_gradients]).mean(dim = 0) 
								for k in meta_gradients[0].keys()}

		'''
		Using hooks to replace the gradient of the meta-model parameters manually.
		https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d has a great
		explanation regarding this.
		'''
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