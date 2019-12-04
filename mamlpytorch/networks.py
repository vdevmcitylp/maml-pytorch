

import torch.nn as nn
import torch.nn.functional as F


class OmniglotCNNModel(nn.Module):

	def __init__(self):
		
		super(OmniglotCNNModel, self).__init()

		

	def forward(self, x, weights = None):
		pass









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

	def functional_forward(self, x, weights):

		x = F.linear(x, weights['hidden_1.weight'], weights['hidden_1.bias'])
		x = F.relu(x)
		# x = F.linear(x, weights[hidden_2.weight], weights[hidden_2.bias])
		output = F.linear(x, weights['output.weight'], weights['output.bias'])

		return output

		