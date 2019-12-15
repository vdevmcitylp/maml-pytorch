# maml-pytorch

This repository is an implementation of [Model Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400) by Finn et al. in PyTorch.

Currently, the code supports first-order MAML (FOMAML) & the [Omniglot](http://www.sciencemag.org/content/350/6266/1332.short) dataset.

### Requirements

- Python 3.5.2
- Pytorch 1.3.1
- Torchmeta 1.2.1

### Acknowledgments

[This](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d) blog post by [Oscar Knagg](https://github.com/oscarknagg) helped me immensely.

<!-- [Add Tensorboard screenshot>] -->

### Usage

#### Setup hyperparameters in config.yml.

```yaml
dataset: omniglot
num_ways: 5
num_shots: 1
meta:
  training_iterations: 10000
  batch_size: 16
  lr: 1.0e-3
inner:
  batch_size: 1
  training_iterations: 1
  lr: 0.4
logs:
  val_interval: 100
  writer_interval: 100
device: 'cpu'               # or 'gpu'
```

#### Run main.py.
	
	python main.py

#### [RECOMMENDED]
To check tensorboard, open up a command prompt and enter,
	
	tensorboard --logdir=tensorboard-runs/

You can select the run ID from the list on the bottom left.