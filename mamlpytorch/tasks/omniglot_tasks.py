
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

class OmniglotTaskDistribution:

	def __init__(self, 
				num_ways,
				num_shots,
				meta_split,
				num_test_shots = 3):
		
		self.num_ways = num_ways
		self.num_shots = num_shots
		self.num_test_shots = num_test_shots
		self.meta_split = meta_split

		self.dataset = omniglot('data', 
								ways = self.num_ways, 
								shots = self.num_shots, 
								test_shots = self.num_test_shots, 
								meta_split = self.meta_split, 
								download = True)

	def sample_batch(self, batch_size):

		dataloader = BatchMetaDataLoader(self.dataset, batch_size = batch_size, num_workers = 4)
		return iter(dataloader).next()
