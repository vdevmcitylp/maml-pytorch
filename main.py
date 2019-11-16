
'''
Require: 
	p(T): Distribution over tasks
	Sinusoidal Regression:
		amplitude: [0.1, 5.0]
		phase: [0, pi]
		p(x): [-5.0, 5.0]
		
		Loss: MSE
		Model: NN [1, ReLU(40), ReLU(40), 1]
		
		1 inner gradient update
		K = 10

	alpha, beta: Step size hyperparameters (Learning Rates)
		alpha: 0.01
		beta: Adam defaults

Baselines:
	Training one network to regress to different random sinusoid functions
	At meta-test, fine-tune with K points and automatically tuned step size (?)

	An oracle which receives true amplitude and phase as input

'''
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def main():

	if args.dataset == "sinusoid":

		metatrain_tasks, metatest_tasks = create_sinusoid_tasks()
		
		model = make_sinusoid_model()

		inner_optimizer = optim.Adam(model.parameters(), lr = args.inner_lr)
		loss_function = nn.MSELoss()
		metrics = []

	metalearner = MAML(args, config).to(device)
	
	else:
		print("ERROR: training task not recognized [", args.dataset, "]")
    	sys.exit()

    if args.metamodel == 'maml':

    	meta_optimizer = optim.GradientDescent()
    	metalearner = MAMLMetaLearner()


	for meta_iter in range(args.num_metatraining_iterations):

		# Fetch meta-batch

		for task in meta_batch:
			pass

		if meta_iter % args.test_every_k_iterations == 0:
			pass

		if meta_iter % args.save_every_k_iterations == 0:
			pass


if __name__ == '__main__':
	
	# Setup argparse
	parser = argparse.ArgumentParser()


	args = parser.parse_args()

	# Call main()
	
	# Set seeds
	# Set device


	# Setup dataset

	# Start meta-train epoch
	# 	Fetch meta-datasets
	# 	Start inner loop
	'''



