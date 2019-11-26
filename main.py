
import os
import time
import pdb
import copy
import yaml
import argparse
import random
import shutil

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from mamlpytorch.tasks.sinusoid_tasks import SinusoidTaskDistribution
from mamlpytorch.networks import SinusoidModel
from mamlpytorch.metalearners.maml import MAMLMetaLearner


def main(cfg, run_id):

	np.random.seed(1)
	torch.manual_seed(1)

	writer = SummaryWriter(log_dir = 'tensorboard-runs/{}/'.format(run_id))

	dataset = cfg.get('dataset', 'sinusoid')

	if dataset == 'sinusoid':
		
		model = SinusoidModel()
		task_distribution = SinusoidTaskDistribution()
		loss_function = nn.MSELoss()

	meta_optimizer = optim.Adam(model.parameters(), lr = cfg['meta']['lr'])

	# start = time.time()

<<<<<<< HEAD
	meta_model = MAMLMetaLearner(model, 
							task_distribution, 
							meta_optimizer, 
							cfg['meta']['batch_size'], 
							cfg['inner']['lr'], 
							cfg['inner']['batch_size'],
							loss_function, 
							order = 1)
	
	meta_test_task = meta_model.task_distribution.sample_batch(batch_size = 1)[0]
	x_query, y_query = meta_test_task.sample_batch(batch_size = cfg['inner']['batch_size'])
	x_support, y_support = meta_test_task.sample_batch(batch_size = cfg['inner']['batch_size'])
=======
		for i, task in enumerate(tasks):
			
			'''
			Step 5
			'''
			x_support, y_support = task.sample_batch(batch_size = cfg['inner']['batch_size'])
			
			'''
			Step 6
			'''
			task_adapted_weights = OrderedDict(model.named_parameters())
			y_support_pred = model.functional_forward(x_support, task_adapted_weights)
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
>>>>>>> f4857dc1587578f6515d2333fe743b6a14e57b8f

	for meta_iter in range(cfg['meta']['training_iterations']):
		
		meta_train_loss = meta_model.train()

		'''
		Meta-Testing
		'''
		if (meta_iter) % cfg['logs']['test_interval'] == 0:
			meta_test_loss = meta_model.test(x_query, y_query, x_support, y_support)
			# fine_tune_loss = fine_tune_model(task)
			writer.add_scalar('Loss/MetaTest', meta_test_loss.item(), meta_iter)

		'''
		Logging Information
		'''
		if (meta_iter + 1) % cfg['logs']['writer_interval'] == 0:
			writer.add_scalar('Loss/MetaTrain', meta_train_loss.item() / cfg['meta']['batch_size'], 
				meta_iter)
		
		if meta_iter % cfg['logs']['save_interval'] == 0:
			if meta_iter == 0:
				best_meta_train_loss = meta_train_loss.item()
				torch.save(model.state_dict(), 'runs/{}/model.pth'.format(run_id))
				continue
			
			if meta_train_loss.item() < best_meta_train_loss:
				best_meta_train_loss = meta_train_loss.item()
				torch.save(model.state_dict(), 'runs/{}/model.pth'.format(run_id))

	# print(time.time() - start)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description = "config")
	parser.add_argument('--config', 
					nargs = '?', 
					type = str, 
					default = 'config.yml')

	args = parser.parse_args()

	with open(args.config) as f_in:
		cfg = yaml.safe_load(f_in)
	
	run_id = random.randint(1, 100000)
	print('Run ID: {}'.format(run_id))

	if not os.path.exists('runs/{}'.format(run_id)):
		os.makedirs('runs/{}'.format(run_id))

	shutil.copy('config.yml', 'runs/{}/config.yml'.format(run_id))

<<<<<<< HEAD
	main(cfg, run_id)
=======
	main(cfg)
>>>>>>> f4857dc1587578f6515d2333fe743b6a14e57b8f
