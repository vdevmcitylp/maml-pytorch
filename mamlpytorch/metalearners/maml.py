'''
Step numbers correspond to the algorithm for supervised learning in the paper (https://arxiv.org/abs/1703.03400)
'''

import torch

from collections import OrderedDict


class MAMLMetaLearner:
	
	def __init__(self,
			model,
			task_distribution,
			meta_optimizer,
			meta_batch_size,
			inner_lr,
			inner_training_iterations,
			inner_batch_size,
			loss_function,
			order):
		
		print('MAML meta-learner initialized!')

		self.meta_optimizer = meta_optimizer
		self.meta_batch_size = meta_batch_size
		self.task_distribution = task_distribution
		self.inner_batch_size = inner_batch_size
		self.inner_lr = inner_lr
		self.inner_training_iterations = inner_training_iterations
		self.model = model
		self.loss_function = loss_function
		self.order = order # 1 or 2


	def train(self):

		'''
		Step 3: Sample tasks from distribution
		'''
		tasks = self.task_distribution.sample_batch(batch_size = self.meta_batch_size)
		
		meta_loss = 0.
		task_query_gradients = []
		
		for i, task in enumerate(tasks):

			'''
			Step 5
			'''
			x_support, y_support = task.sample_batch(batch_size = self.inner_batch_size)

			task_adapted_weights = self.inner_train(x_support, y_support)
			
			'''
			Step 8
			'''
			x_query, y_query = task.sample_batch(batch_size = self.inner_batch_size)
			
			task_query_gradient, task_query_loss = self.get_query_gradient_loss(x_query, y_query, \
																				task_adapted_weights)
	
			task_query_gradients.append(task_query_gradient)
			meta_loss += task_query_loss
	
			meta_gradient = self.get_meta_gradient(task_query_gradients)

			self.update(meta_gradient, meta_loss)

		return meta_loss

	def inner_train(self, x_support, y_support):
		'''
		Step 6
		'''
		task_adapted_weights = OrderedDict(self.model.named_parameters())
		
		for inner_iter in range(self.inner_training_iterations):

			y_support_pred = self.model.functional_forward(x_support, task_adapted_weights)
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
		y_query_pred = self.model.functional_forward(x_query, task_adapted_weights)
		task_query_loss = self.loss_function(y_query_pred, y_query)
		task_query_gradient = torch.autograd.grad(task_query_loss, task_adapted_weights.values(), create_graph = False)
			
		'''
		Naming the gradients
		'''
		task_query_gradient = {name: g for ((name, _), g) in zip(task_adapted_weights.items(), task_query_gradient)}
		
		return task_query_gradient, task_query_loss

	def get_meta_gradient(self, task_query_gradients):
		
		'''
		Step 10.1: Calculating meta-gradient
		'''
		meta_gradient = {k: torch.stack([grad[k] for grad in task_query_gradients]).mean(dim = 0) 
								for k in task_query_gradients[0].keys()}

		return meta_gradient

	def update(self, meta_gradient, meta_loss):

		'''
		Step 10.2: Updating the model weights
		'''

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
	
			dummy_pred = self.model(torch.zeros(1, 1))
			dummy_loss = self.loss_function(dummy_pred, torch.zeros(1, 1))
			
			'''
			Replacing gradient of every parameter in the meta-model using a backward hook
			'''
			dummy_loss.backward()
			self.meta_optimizer.step()

			for h in hooks:
				h.remove()

	def test(self, x_query, y_query, x_support, y_support):
		'''
		Perform meta-testing
		See the effectiveness of the meta-learning procedure by performing k-shot testing on a new task
		'''

		task_adapted_weights = self.inner_train(x_support, y_support)
				
		_, task_query_loss = self.get_query_gradient_loss(x_query, y_query, task_adapted_weights)

		return task_query_loss
