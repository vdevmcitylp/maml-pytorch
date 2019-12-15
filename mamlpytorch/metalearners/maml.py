'''
Step numbers correspond to the algorithm for supervised learning in the paper (https://arxiv.org/abs/1703.03400)
'''

import logging

import torch
import torch.nn.functional as F

from collections import OrderedDict

import pdb

LOGGER = logging.getLogger(__name__)


class MAMLMetaLearner:
	
	def __init__(self,
			model, 
			meta_optimizer,
			meta_batch_size,
			inner_lr,
			inner_training_iterations,
			inner_batch_size,
			loss_function,
			device = 'cpu',
			order = 1):
		
		LOGGER.info('MAML meta-learner initialized!')

		self.meta_optimizer = meta_optimizer
		self.meta_batch_size = meta_batch_size
		self.inner_batch_size = inner_batch_size
		self.inner_lr = inner_lr
		self.inner_training_iterations = inner_training_iterations
		self.model = model.to(device)
		self.loss_function = loss_function
		self.order = order # 1
		self.device = device


	def train(self, meta_train_task_distribution):

		'''
		Step 3: Sample tasks from distribution
		'''
		tasks = meta_train_task_distribution.sample_batch(batch_size = self.meta_batch_size)

		meta_loss = 0.
		task_query_gradients = []
		
		x_support_batch, y_support_batch = tasks['train']
		x_query_batch, y_query_batch = tasks['test']

		for x_support, y_support, x_query, y_query in zip(x_support_batch, y_support_batch, 
															x_query_batch, y_query_batch):

			x_support, y_support = x_support.to(self.device), y_support.to(self.device)
			x_query, y_query = x_query.to(self.device), y_query.to(self.device)

			task_adapted_weights = self.inner_train(x_support, y_support)
				
			task_query_gradient, task_query_loss, _ = self.get_query_gradient_loss(x_query, y_query, \
																				task_adapted_weights)
	
			task_query_gradients.append(task_query_gradient)
			meta_loss += task_query_loss
	
			'''
			Step 10.1: Calculating meta-gradient
			'''
			meta_gradient = self.get_meta_gradient(task_query_gradients)

			'''
			Step 10.2: Updating the model weights
			'''
			self.update(meta_gradient, meta_loss)

		return meta_loss

	def inner_train(self, x_support, y_support):
		'''
		Step 6
		'''
		task_adapted_weights = OrderedDict(self.model.named_parameters())
		
		for inner_iter in range(self.inner_training_iterations):

			y_support_pred = self.model.functional_forward(x_support, weights = task_adapted_weights)
			task_support_loss = self.loss_function(y_support_pred, y_support)
			task_support_gradient = torch.autograd.grad(task_support_loss, task_adapted_weights.values())

			task_adapted_weights = self._inner_update(task_adapted_weights, task_support_gradient)

		return task_adapted_weights

	def _inner_update(self, task_adapted_weights, task_support_gradient):
		'''
		Step 7
		'''
		task_adapted_weights = OrderedDict(
			(name, param - self.inner_lr * grad)
			for ((name, param), grad) in zip(task_adapted_weights.items(), task_support_gradient))

		return task_adapted_weights

	def get_query_gradient_loss(self, x_query, y_query, task_adapted_weights):		
		'''
		Save gradients for each task
		'''
		y_query_logit = self.model.functional_forward(x_query, task_adapted_weights)
		task_query_loss = self.loss_function(y_query_logit, y_query)
		task_query_gradient = torch.autograd.grad(task_query_loss, task_adapted_weights.values(), create_graph = False)
			
		'''
		Naming the gradients
		'''
		task_query_gradient = {name: g for ((name, _), g) in zip(task_adapted_weights.items(), task_query_gradient)}
		
		return task_query_gradient, task_query_loss, y_query_logit

	def get_meta_gradient(self, task_query_gradients):
		
		meta_gradient = {k: torch.stack([grad[k] for grad in task_query_gradients]).mean(dim = 0) 
								for k in task_query_gradients[0].keys()}

		return meta_gradient

	def update(self, meta_gradient, meta_loss):

		def replace_grad(param_grad, param_name):
			def replace_grad_(module):
				return param_grad[param_name]

			return replace_grad_

		'''
		For 1st order MAML (FOMAMAL)
		'''
		if self.order == 1:

			hooks = []
			for name, param in self.model.named_parameters():
				hooks.append(
					param.register_hook(replace_grad(meta_gradient, name)))

			self.meta_optimizer.zero_grad()
	
			dummy_pred = self.model(torch.zeros(1, 1, 28, 28)) # TD: Define according to dataset
			dummy_loss = self.loss_function(dummy_pred, torch.LongTensor([1]))
			
			'''
			Replacing gradient of every parameter in the meta-model using a backward hook
			'''
			dummy_loss.backward()
			self.meta_optimizer.step()

			for h in hooks:
				h.remove()

	def validate(self, meta_val_task_distribution):
		'''
		Perform meta-validation
		Note: The same set of images are not being used each time for validation
		The set of classes are the same though
		'''
		val_tasks = meta_val_task_distribution.sample_batch(batch_size = self.meta_batch_size)

		x_support_batch, y_support_batch = val_tasks['train']
		x_query_batch, y_query_batch = val_tasks['test']

		correct_predictions = 0
		total_predictions = 0 
		meta_val_loss = 0.

		for x_support, y_support, x_query, y_query in zip(x_support_batch, y_support_batch, 
															x_query_batch, y_query_batch):

			x_support, y_support = x_support.to(self.device), y_support.to(self.device)
			x_query, y_query = x_query.to(self.device), y_query.to(self.device)

			task_adapted_weights = self.inner_train(x_support, y_support)
				
			_, task_query_loss, y_query_logit = self.get_query_gradient_loss(x_query, y_query, \
																				task_adapted_weights)

			meta_val_loss += task_query_loss

			with torch.no_grad():
				y_query_pred = F.softmax(y_query_logit, dim = 1).argmax(dim = 1)
				correct_predictions += torch.eq(y_query_pred, y_query).sum().item()
				total_predictions += len(y_query)

		meta_val_accuracy = correct_predictions / total_predictions
		
		return meta_val_loss, meta_val_accuracy

	def test(self, best_model, meta_test_task_distribution):

		best_model.eval()

		test_tasks = meta_test_task_distribution.sample_batch(batch_size = self.meta_batch_size)

		x_support_batch, y_support_batch = test_tasks['train']
		x_query_batch, y_query_batch = test_tasks['test']

		correct_predictions = 0
		total_predictions = 0 
		meta_test_loss = 0.

		for x_support, y_support, x_query, y_query in zip(x_support_batch, y_support_batch, 
															x_query_batch, y_query_batch):

			x_support, y_support = x_support.to(self.device), y_support.to(self.device)
			x_query, y_query = x_query.to(self.device), y_query.to(self.device)

			task_adapted_weights = self.inner_train(x_support, y_support)
				
			_, task_query_loss, y_query_logit = self.get_query_gradient_loss(x_query, y_query, \
																				task_adapted_weights)

			meta_test_loss += task_query_loss

			with torch.no_grad():
				y_query_pred = F.softmax(y_query_logit, dim = 1).argmax(dim = 1)
				correct_predictions += torch.eq(y_query_pred, y_query).sum().item()
				total_predictions += len(y_query)

		meta_test_accuracy = correct_predictions / total_predictions
		
		return meta_test_accuracy