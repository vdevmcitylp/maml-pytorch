
import pdb
import copy

import torch
import torch.nn as nn
import torch.optim as optim


from mamlpytorch.tasks.sinusoid_tasks import SinusoidTaskDistribution
from mamlpytorch.networks import SinusoidModel


'''
Theta: Meta-parameters
Theta': Task-parameters
'''
model = SinusoidModel()

inner_lr = 1e-4
meta_lr = 1e-4

meta_optimizer = optim.Adam(model.parameters(), lr = meta_lr)

num_tasks = 10
task_distribution = SinusoidTaskDistribution()
num_datapoints = 5

loss_function = nn.MSELoss()

num_meta_iterations = 10000
num_inner_training_iterations = 1
# pdb.set_trace()

for meta_iter in range(num_meta_iterations):
	'''
	Sample batch of tasks: T = [T1, T2, ... Tn]
	'''
	tasks = task_distribution.sample_batch(num_tasks = num_tasks)
	
	all_task_losses = 0

	for i, task in enumerate(tasks):
		'''
		Sample K data-points [(x_1, y_1), (x_2, y_2), ..., (x_K, y_K)] from task
		'''
		x_support, y_support = task.sample_batch(num_datapoints = num_datapoints)
		x_query, y_query = task.sample_batch(num_datapoints = num_datapoints)

		# task_model = SinusoidModel()
		# task_model.load_state_dict(model.state_dict())
		
		task_model = copy.deepcopy(model)
		inner_optimizer = optim.Adam(task_model.parameters(), lr = inner_lr)

		for inner_iteration in range(num_inner_training_iterations):

			inner_train_loss = loss_function(task_model(x_support), y_support)
			inner_train_loss.backward(create_graph = False)
			inner_optimizer.step()

		inner_test_loss = loss_function(task_model(x_query), y_query)
		all_task_losses += inner_test_loss
	if meta_iter % 500 == 0:
		print("Meta-iter: {} All Task Loss - {}".format(meta_iter, all_task_losses))

	meta_optimizer.zero_grad()
	all_task_losses.backward()
	meta_optimizer.step()

			# inner_gradient = torch.autograd.grad(inner_train_loss, task_model.parameters())
			# for weight, gradient in zip(task_model.parameters(), inner_gradient):
			# 	weight = weight - (inner_lr * gradient)
		
		# task_loss.backward() -> delL(theta)/del(theta)
	# 	inner_optimizer.step() # theta_i' = theta - inner_lr * (delta(L(theta)))
		
	# 	Sample K data-points [(x_1, y_1), (x_2, y_2), ..., (x_K, y_K)] from task
	# 	for meta-gradient update
		
	# 	task_test_pred = task_model(task_test_x) # With adapted parameters peformed by inner_optimizer.step()
	# 	task_test_loss = loss_fun(task_test_pred, task_test_y)
	# 	task_losses += task_test_loss
