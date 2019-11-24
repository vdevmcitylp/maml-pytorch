

class MAMLMetaLearner:
	
	def __init__(self):
		print('MAML meta-learner initialized!')

		self.meta_optimizer = meta_optimizer
		self.meta_lr = meta_lr
		self.meta_batch_size = meta_batch_size
		# self.meta_train_iter = 

		self.model = model

	def train():
		'''
		Perform one meta-train iteration
		'''

		tasks = task_distribution.sample_batch(batch_size = cfg['meta']['batch_size'])
		
		meta_loss = 0.
		meta_gradients = []

		for i, task in enumerate(tasks):
			
			'''
			Step 5
			'''
			x_support, y_support = task.sample_batch(batch_size = cfg['inner']['batch_size'])
			
			'''
			Step 6
			'''
			task_adapted_weights = OrderedDict(model.named_parameters())
			y_support_pred = model.functional_forward(x_support, fast_weights)
			task_support_loss = loss_function(y_support_pred, y_support)
			task_support_gradient = torch.autograd.grad(task_support_loss, task_adapted_weights.values())

			'''
			Step 7
			'''
			task_adapted_weights = OrderedDict(
				(name, param - inner_lr * grad)
				for ((name, param), grad) in zip(task_adapted_weights.items(), task_support_gradient))

			'''
			Step 8
			'''
			x_query, y_query = task.sample_batch(batch_size = cfg['inner']['batch_size'])
			
			'''
			Save gradients for each task
			'''
			y_query_pred = model.functional_forward(x_query, task_adapted_weights)
			task_query_loss = loss_function(y_query_pred, y_query)
			task_query_gradient = torch.autograd.grad(task_query_loss, task_adapted_weights.values(), create_graph = False)
			
			'''
			Naming the gradients
			'''
			task_query_gradient = {name: g for ((name, _), g) in zip(task_adapted_weights.items(), task_query_gradient)}
			
			meta_gradients.append(task_query_gradient) 

			meta_loss += task_query_loss
		
		if (meta_iter + 1) % writer_interval == 0:
			writer.add_scalar('Loss/QuerySet', meta_loss.item() / cfg['meta']['batch_size'], 
				meta_iter)
		
		'''
		Step 10
		'''
		meta_gradient = {k: torch.stack([grad[k] for grad in meta_gradients]).mean(dim = 0) 
								for k in meta_gradients[0].keys()}
		
		meta_update()	

		define_hooks(model)

		hooks = []
		for name, param in model.named_parameters():
			hooks.append(
				param.register_hook(replace_grad(meta_gradient, name)))

		meta_optimizer.zero_grad()
		
		dummy_pred = model(torch.zeros(1, 1))
		dummy_loss = loss_function(dummy_pred, torch.zeros(1, 1))
		'''
		Replacing gradient of every parameter in the meta-model using a backward hook
		'''
		dummy_loss.backward()
		meta_optimizer.step()

		for h in hooks:
			h.remove()


	def test():
		'''
		Perform meta-testing
		'''

	# def update():

