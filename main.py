import argparse
import sys
sys.path.append(".")
import crown
import models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils

if __name__ == "__main__":

	# parser = argparse.ArgumentParser()
	# parser.add_argument('--model', required=True,
	# 	choices=[
     #        ''
	# 		# Exeriment 2 (minimum adversarial distortion)
	# 		# ------------------------------------
	# 		'mnist_cnn_small',
	# 		'mnist_cnn_wide_1',
	# 		'mnist_cnn_wide_2',
	# 		'mnist_cnn_wide_4',
	# 		'mnist_cnn_wide_8',	'MLP_9_500',
	# 		'mnist_cnn_deep_1',	'MLP_9_100',
	# 		'mnist_cnn_deep_2',	'MLP_2_100',
     #        'cifar_cnn_small','cifar_cnn_small',
     #        'cifar_cnn_wide_1','cifar_cnn_wide_2',
     #        'cifar_cnn_wide_4'
	# 		])
    #
	# parser.add_argument('--epsilon', type=float, help='The maximum allowed l_inf perturbation of the attack during test time.')
	# parser.add_argument('--training-mode', choices=['ADV','LPD','NOR'], help='The training mode of the model. \
	# 				- ADV: The PGD training of Madry et. al. \
	# 				- LPD: dual formulation training of Wong and Kolter \
	# 				- NOR: regular XEntropy training \
	# 				If the model name starts with "ADV", "LPD", or "NOR", this argument is ignored; the training mode in \
	# 				this case is infered from the model name.')
	# parser.add_argument('--batch-size', type=int, default=100)
	# parser.add_argument('--lp-greedy', action='store_true', help='A flag to include lp-greedy as an evaluation method \
	# 															in addition to pgd and normal errors.')
	# parser.add_argument('--data_dir', type=str, default='data',
	# 	help='Directory where MNIST dataset is stored.')
	# args = parser.parse_args()
    #
	# if args.model.startswith(('ADV','LPD','NOR')):
	# 	args.training_mode = args.model.split('_')[0]
	###########################
    # Loading the model
    ###########################
	print('--------------------------------------------------------------')
	#print('Model: {} | Training mode: {} | Testing epsilon: {}'.format(args.model, args.training_mode, args.epsilon))
	print('Model: {} | Training mode: {} | Testing epsilon: {}'.format('cifar_cnn_small', 'NOR', 0.001))
	print('--------------------------------------------------------------')
	#model = models.get_model(args.model, args.training_mode)
	model = models.get_model('cifar_cnn_small', 'NOR')
	print(model.eval())
	#_,test_loader = utils.cifar_loaders(args.batch_size, shuffle_test=False)
	_,test_loader = utils.cifar_loaders(8, shuffle_test=False)
	inputsize = (3,32,32)
	imp, err, aferr,pgd_err,pgd_aferr = crown.replace_bound(inputsize, model,test_loader,eps = 0.001,pgd=False,verbose = 10)







