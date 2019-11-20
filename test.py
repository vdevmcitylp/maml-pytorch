
import numpy as np

from mamlpytorch.networks import SinusoidModel
from mamlpytorch.tasks.sinusoid_tasks import SinusoidTask

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


from itertools import islice

import pdb

stask = SinusoidTask()
sloader = DataLoader(stask, batch_size = 4, shuffle = True)

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

model = SinusoidModel().to(device)

loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

print(len(sloader))
for epoch in range(1):

	avg_loss = 0.
	for i, batch in enumerate(islice(sloader, 0, 1)):

		# print(i, batch)
		# print(batch[0])

		x_batch, y_batch = batch[0].to(device), batch[1].to(device)

		pred_batch = model(batch[0].float())
		# print(pred_batch)

		loss_batch = loss(y_batch.float(), pred_batch.float())
		avg_loss += loss_batch.item()
		gradients = torch.autograd.grad(loss_batch, model.parameters(), create_graph = True)
		pdb.set_trace()
		print(gradients)
		# loss_batch.backward()
		optimizer.step()

	if epoch % 100 == 0:
		print(avg_loss/len(sloader))



# for meta_iter in range(num_meta_iterations):

# 	for task in metatrain_tasks:

# 		inner_task_weights = OrderedDict(model.parameters())

# 		for i, batch in enumerate(task_loader):
# 			pred = model.functional_forward(x_task_train, inner_task_weights)
# 			loss = loss_function(pred, y)
			# gradients = torch.autograd.grad(loss, inner_task_weights.values(), create_graph = True)


# 			# Update inner_task_weights
			