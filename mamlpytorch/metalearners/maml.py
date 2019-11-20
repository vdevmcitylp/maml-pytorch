

class MAMLMetaLearner:
	
	def __init__(self):
		print('MAML meta-learner initialized!')

		self.meta_optimizer = meta_optimizer
		self.meta_lr = meta_lr
		self.meta_batch_size = meta_batch_size

	# Meta-training
	def train():
		'''
		This function implements one meta-iteration
		'''
		for task in metatrain_tasks:

			# Sample K data-points from task
			# Initialize PyTorch dataloader object

			# Perform a few iteration for the inner task parameters
			task.train(model, dataloader, num_inner_training_iterations, inner_batch_size)

			# Test to track metric for that task
			task.test(model, inner_batch_size)

  	# Meta-testing
	def test():