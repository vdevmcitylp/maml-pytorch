

import torch.nn as nn
import torch.nn.functional as F

class SinusoidModel(nn.Module):

	def __init__(self):

		super(SinusoidModel, self).__init__()
		
		self.hidden_1 = nn.Linear(1, 40)
		# self.hidden_2 = nn.Linear(40, 40)
		self.output = nn.Linear(40, 1)

	def forward(self, x):

		x = self.hidden_1(x)
		x = F.relu(x)
		# x = self.hidden_2(x)
		# x = F.relu(x)
		output = self.output(x)

		return output

	'''
	This could have been defined anywhere, makes OOP-sense to put it here
	'''
	def functional_forward(self, x, weights):

		x = F.linear(x, weights['hidden_1.weight'], weights['hidden_1.bias'])
		x = F.relu(x)
		# x = F.linear(x, weights[2], weights[3])
		output = F.linear(x, weights['output.weight'], weights['output.bias'])

		return output

		