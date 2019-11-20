
import argparse

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

from mamlpytorch.tasks.sinusoid_tasks import create_sinusoid_tasks
from mamlpytorch.networks import SinusoidModel
from mamlpytorch.metalearners.maml import MAMLMetaLearner


def main():

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if args.dataset == "sinusoid":

		metatrain_task_distribution, metatest_task_distribution = \
							create_sinusoid_tasks(min_amplitude = 0.1, 
													max_amplitude = 5.0,
													min_phase = 0.0,
													max_phase = 2 * np.pi,
													min_x = -5.0,
													max_x = 5.0,
													num_training_samples = args.num_train_samples_per_class,
													num_test_samples = args.num_test_samples_per_class,
													num_test_tasks = 100,
													meta_batch_size = args.meta_batch_size)
		
		model = SinusoidModel().to(device)

		inner_optimizer = optim.Adam(model.parameters(), lr = args.inner_lr)
		loss_function = nn.MSELoss()
		metrics = []
	
	else:
		print("ERROR: training task not recognized [", args.dataset, "]")
		sys.exit()

	if args.metamodel == 'maml':

		meta_optimizer = optim.Adam(lr = args.meta_lr)
		metalearner = MAMLMetaLearner()

	metabatch_results = [] # Store loss from each task
	for meta_iter in range(args.num_metatraining_iterations):

		# Sample a new set of tasks for every meta-iteration
		# meta_batch should be a list of SinusoidTask objects
		# Create 'batch_size' PyTorch dataset objects
			# Initialize a dataloader for each dataset 

		metatrain_tasks, metatest_tasks = create_sinusoid_tasks(min_amplitude = 0.1, 
													max_amplitude = 5.0,
													min_phase = 0.0,
													max_phase = 2 * np.pi,
													min_x = -5.0,
													max_x = 5.0,
													num_training_samples = args.num_train_samples_per_class,
													num_test_samples = args.num_test_samples_per_class,
													num_test_tasks = 100,
													meta_batch_size = args.meta_batch_size)

		metalearner.train()		

		for task in metatrain_tasks:
			# print("Amplitude: {}, Phase: {}".format(task.X.shape, task.y.shape))
			task.fit_n_iterations(model, inner_optimizer, loss_function, args.num_inner_training_iterations, \
				args.inner_batch_size)

			# metabatch_results.append(metalearner.task_end(task))

		if meta_iter % args.test_every_k_iterations == 0:
			pass

		if meta_iter % args.save_every_k_iterations == 0:
			pass

	# metalearner.update()

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	
	# Dataset and model options
	parser.add_argument('--dataset', default = 'sinusoid')
	parser.add_argument('--metamodel', default = 'maml')

	parser.add_argument('--num_output_classes', default = 5, help = 'number of classes used in classification \
		(e.g. 5-way classification).')
	parser.add_argument('--num_train_samples_per_class', default = 5, help = 'number of samples per class used in \
		classification (e.g. 5-shot classification).')
	parser.add_argument('--num_test_samples_per_class', default = 15, help = 'number of samples per class used in \
		testing (e.g., evaluating a model trained on k-shots, on a different set of samples).')
	
	# Meta-training options
	parser.add_argument('--num_metatraining_iterations', default = 1000)
	parser.add_argument('--meta_batch_size', default = 5, help = 'meta-batch size: number of tasks sampled at \
		each meta-iteration.')
	parser.add_argument('--meta_lr', default = 0.001, help = 'learning rate of the meta-optimizer')

	# Inner-training options
	parser.add_argument('--num_inner_training_iterations', default = 5, help = 'number of gradient descent steps \
		to perform for each task in a meta-batch (inner steps).')
	parser.add_argument('--inner_batch_size', default = -1, help = 'batch size: number of task-specific points \
		sampled at each inner iteration. If <0, then it defaults to num_train_samples_per_class*num_output_classes.')
	parser.add_argument('--inner_lr', default = 0.001, help = 'learning rate of the inner optimizer. \
		Default 0.01 for FOMAML, 1.0 for Reptile')

	# Logging, saving, and testing options
	parser.add_argument('--save_every_k_iterations', default = 1000, help = 'the model is saved every k iterations.')
	parser.add_argument('--test_every_k_iterations', default = 100, help = 'the performance of the model is evaluated every \
		k iterations.')
	parser.add_argument('--model_save_filename', default = 'saved/model.h5', help = 'path + filename where to save the \
		model to.')

	parser.add_argument('--seed', default = '100', type = int, help = 'random seed.')

	args = parser.parse_args()

	main()
	


