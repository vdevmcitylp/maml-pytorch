

class Task:

	def __init__(self):
		print('Task Superclass')

	def fit_n_iterations(self, model, num_iterations, batch_size):

		for iteration in range(num_iterations):
			
			for i, batch in enumerate(dataloader):
				x_batch, y_batch = batch[0].to(device), batch[1].to(device)

				pred_batch = model(batch[0].float())
				# print(pred_batch)

				loss_batch = loss(y_batch.float(), pred_batch.float())
				avg_loss += loss_batch.item()

				loss_batch.backward()
				optimizer.step()

			if iteration % 100 == 0:
				print(avg_loss/len(sloader))			
