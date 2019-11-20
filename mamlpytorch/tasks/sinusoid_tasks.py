
import numpy as np
import pdb
import torch
# fom torch.utils.data import Dataset
# from mamlpytorch.core.task import Task

class SinusoidTask:
	
	def __init__(self, 
		amplitude,
		phase,
		min_x,
		max_x):
		
		self.amplitude = amplitude
		self.phase = phase
		self.min_x = min_x
		self.max_x = max_x

	def sample_batch(self, num_datapoints = 5):

		# Returns a batch of datapoints from the given amplitude & phase

		x = np.random.uniform(self.min_x, self.max_x, size = (num_datapoints, 1))
		y = self.amplitude * np.sin(x + self.phase)

		x = torch.FloatTensor(x)
		y = torch.FloatTensor(y)

		return x, y


class SinusoidTaskDistribution:

	def __init__(self, 
				min_amplitude = 0.1,
				max_amplitude = 5.0,
				min_phase = 0.0,
				max_phase = np.pi,
				min_x = -5.0,
				max_x = 5.0):

		self.min_amplitude = min_amplitude
		self.max_amplitude = max_amplitude
		self.min_phase = min_phase
		self.max_phase = max_phase
		self.min_x = min_x
		self.max_x = max_x

	def sample_task(self):
		'''
		Creates a SinusoidTask object	
		'''
		amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
		phase = np.random.uniform(self.min_phase, self.max_phase)

		return SinusoidTask(amplitude, phase, self.min_x, self.max_x)

	def sample_batch(self, num_tasks):
		'''
		Returns a list of SinusoidTask objects
		'''
		tasks = []
		for task_index in range(num_tasks):
			tasks.extend([self.sample_task()])

		return tasks

if __name__ == '__main__':
	
	task_distribution = SinusoidTaskDistribution()
	num_tasks = 2
	tasks = task_distribution.sample_batch(num_tasks)

	pdb.set_trace()
	for task in tasks:
		x, y = task.sample_batch()
		