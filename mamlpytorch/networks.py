
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

'''
Slight modifications to code taken from https://github.com/oscarknagg/few-shot/blob/master/few_shot/models.py
'''

def conv_block(in_channels, out_channels):
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )

def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
                          bn_weights, bn_biases) -> torch.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.
    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class OmniglotCNNModel(nn.Module):

	def __init__(self, num_ways):
		
		super(OmniglotCNNModel, self).__init__()

		self.conv_block1 = conv_block(1, 64)
		self.conv_block2 = conv_block(64, 64)
		self.conv_block3 = conv_block(64, 64)
		self.conv_block4 = conv_block(64, 64)

		self.logits = nn.Linear(64, num_ways)

	def forward(self, x):

		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = self.conv_block3(x)
		x = self.conv_block4(x)

		x = x.view(x.size(0), -1)

		return self.logits(x)

	def functional_forward(self, x, weights):

		for block in [1, 2, 3, 4]:
			x = functional_conv_block(x, weights['conv_block{}.0.weight'.format(block)], 
											weights['conv_block{}.0.bias'.format(block)], 
											weights.get('conv_block{}.1.weight'.format(block)), 
											weights.get('conv_block{}.1.bias'.format(block)))

		x = x.view(x.size(0), -1)

		x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

		return x


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

		