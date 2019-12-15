
import os
import time
import pdb
import copy
import yaml
import argparse
import random
import shutil
import logging

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


logging.getLogger("matplotlib").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

def init_logging():
    logging.basicConfig(format = '[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level = logging.DEBUG)

def main(cfg, run_id):

	np.random.seed(1)
	torch.manual_seed(1)
	
	device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
	LOGGER.info('Training on {}.'.format(cfg['device']))

	writer = SummaryWriter(log_dir = 'tensorboard-runs/{}/'.format(run_id))

	dataset = cfg.get('dataset', 'sinusoid')

	if dataset == 'sinusoid':
		raise NotImplementedError
		# model = SinusoidModel()
		# task_distribution = SinusoidTaskDistribution()
		# loss_function = nn.MSELoss()

	elif dataset == 'omniglot':

		model = OmniglotCNNModel(num_ways = cfg['num_ways'])
		metatrain_task_distribution = OmniglotTaskDistribution(num_ways = cfg['num_ways'], num_shots = cfg['num_shots'], \
																meta_split = 'train')
		metaval_task_distribution = OmniglotTaskDistribution(num_ways = cfg['num_ways'], num_shots = cfg['num_shots'], \
																meta_split = 'val')
		loss_function = nn.CrossEntropyLoss()

	LOGGER.info('Using {} dataset'.format(dataset))

	meta_optimizer = optim.Adam(model.parameters(), lr = cfg['meta']['lr'])

	meta_model = MAMLMetaLearner(model,
							meta_optimizer, 
							cfg['meta']['batch_size'], 
							cfg['inner']['lr'], 
							cfg['inner']['training_iterations'], 
							cfg['inner']['batch_size'],
							loss_function, 
							device,
							order = 1)


	for meta_iter in range(cfg['meta']['training_iterations']):
		
		meta_train_loss = meta_model.train(metatrain_task_distribution)

		'''
		Meta-Validation
		'''
		if (meta_iter) % cfg['logs']['val_interval'] == 0:
			LOGGER.info('Performing meta-validation at meta-iteration: {}'.format(meta_iter))
			meta_val_loss, meta_val_accuracy = meta_model.validate(metaval_task_distribution)
			writer.add_scalar('Loss/MetaVal', meta_val_loss.item(), meta_iter)
			writer.add_scalar('Accuracy/MetaVal', meta_val_accuracy, meta_iter)

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
				LOGGER.info('Saving a better model at meta-iteration: {}'.format(meta_iter))
				best_meta_train_loss = meta_train_loss.item()
				torch.save(model.state_dict(), 'runs/{}/model.pth'.format(run_id))

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description = "config")
	parser.add_argument('--config', 
					nargs = '?', 
					type = str, 
					default = 'config.yml')

	args = parser.parse_args()

	init_logging()

	with open(args.config) as f_in:
		cfg = yaml.safe_load(f_in)
	
	run_id = random.randint(1, 100000)
	LOGGER.info('Run ID: {}'.format(run_id))

	if not os.path.exists('runs/{}'.format(run_id)):
		os.makedirs('runs/{}'.format(run_id))

	shutil.copy('config.yml', 'runs/{}/config.yml'.format(run_id))
	LOGGER.info('Saved configuration file.')

	main(cfg, run_id)
