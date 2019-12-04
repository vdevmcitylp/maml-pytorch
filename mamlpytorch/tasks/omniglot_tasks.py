
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

class OmniglotTaskDistribution:

	def __init__(self, 
				num_ways,
				num_shots,
				num_test_shots,
				meta_train = True):
		
		self.num_ways = num_ways
		self.num_shots = num_shots
		self.num_test_shots = num_test_shots
		self.meta_train = True

		self.dataset = omniglot('data', 
								ways = self.num_ways, 
								shots = self.num_shots, 
								test_shots = self.num_test_shots, 
								meta_train = self.meta_train, 
								download = False)

	def sample_batch(self, batch_size):

		dataloader = BatchMetaDataLoader(self.dataset, batch_size = batch_size, num_workers = 4)
		return dataloader
