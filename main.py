
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
from mamlpytorch.tasks.omniglot_tasks import OmniglotTaskDistribution

from mamlpytorch.networks import SinusoidModel, OmniglotCNNModel

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

	elif dataset == 'omniglot':

		model = OmniglotCNNModel(num_ways = 5)
		task_distribution = OmniglotTaskDistribution(num_ways = 5, num_shots = 1)
		loss_function = nn.CrossEntropyLoss()

	meta_optimizer = optim.Adam(model.parameters(), lr = cfg['meta']['lr'])

	# start = time.time()

	meta_model = MAMLMetaLearner(model, 
							task_distribution, 
							meta_optimizer, 
							cfg['meta']['batch_size'], 
							cfg['inner']['lr'], 
							cfg['inner']['training_iterations'], 
							cfg['inner']['batch_size'],
							loss_function, 
							order = 1)
	
	# Wrap in a function, decide where to place
	meta_test_task = task_distribution.sample_batch(batch_size = 1)
	x_support, y_support = meta_test_task['train']
	x_query, y_query = meta_test_task['test']

	x_support = torch.squeeze(x_support, 0)
	y_support = torch.squeeze(y_support)

	x_query = torch.squeeze(x_query, 0)
	y_query = torch.squeeze(y_query)

	for meta_iter in range(cfg['meta']['training_iterations']):
		
		meta_train_loss = meta_model.train()

		'''
		Meta-Testing
		'''
		if (meta_iter) % cfg['logs']['test_interval'] == 0:
			meta_test_loss, meta_test_accuracy = meta_model.test(x_query, y_query, x_support, y_support) 
			# fine_tune_loss = fine_tune_model(task)
			writer.add_scalar('Loss/MetaTest', meta_test_loss.item(), meta_iter)
			writer.add_scalar('Accuracy/MetaTest', meta_test_accuracy, meta_iter)

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

	main(cfg, run_id)
