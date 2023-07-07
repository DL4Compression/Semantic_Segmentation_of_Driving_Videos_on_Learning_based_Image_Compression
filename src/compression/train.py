import torch, torchvision, os
from torch.utils import data
from torch.backends import cudnn
import functools, operator, numpy as np
from termcolor import colored
import json
cudnn.benchmark = True
from utils.dataloader import IDD
from utils.model import autoencoder
from torchmetrics import PeakSignalNoiseRatio


from tqdm import tqdm as tq
import random


# For Reproducability
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main( args ):
	images_transforms = torchvision.transforms.Compose([ torchvision.transforms.ToTensor()])
	images_transforms_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
	iddtrain = IDD(args.traindata,
		transform_images=images_transforms)
	iddtraindl = data.DataLoader(iddtrain, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
	iddtest = IDD(args.testdata,
		transform_images=images_transforms_test)
	iddtestdl = data.DataLoader(iddtest, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)

	# Model instance with variable number of DownConv blocks
	# args.n_downconvs = 3 corresponds to CVPR paper
	model = autoencoder(args.n_convblocks)
	if torch.cuda.is_available() and args.gpu:
		model = model.cuda()
	optimizer = torch.optim.Adam(model.parameters()) # usual optimizer instance
	schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75) # schedular instance
	psnr_loss = PeakSignalNoiseRatio().cuda()
	# Resume from model
	start_epoch = 0
	# load model file (if present) and load the model,optimizer & schedular states
	model_file = '.'.join([args.model_file_name, 'model'])
	if os.path.exists(os.path.abspath(model_file)):
		loaded_file = torch.load(os.path.abspath(model_file))
		model.load_state_dict(loaded_file['model_state'])
		optimizer.load_state_dict(loaded_file['optim_state'])
		schedular.load_state_dict(loaded_file['schedular_state'])
		start_epoch = loaded_file['epoch'] + 1
	
		print('{tag} resuming from saved model'.format(tag=colored('[Saving]', 'red')));
		del loaded_file
	
	# For logging purpose;
	logg = [] 
	log_file = '.'.join([args.model_file_name, 'log'])
	if os.path.exists(os.path.abspath(log_file)):
		# if .log file exists, open it
		with open(log_file, 'r') as logfile:
			logg = json.load(logfile)
	
	msecrit = torch.nn.MSELoss() # usual MSE loss function

	prev_test_psnr = -np.inf
	for epoch in range(start_epoch, args.epochs):
		# regularly invoke the schedular

		print(f'Epoch:{epoch}/{args.epochs}')

		model.train() # training mode ON
		cnt=0
		avg_train_loss = 0
		avg_train_ssim = 0
		avg_train_psnr = 0

		print("Training Loop!")

		for idx, images in enumerate(tq(iddtraindl)):
			# The data fetch loop
			if torch.cuda.is_available() and args.gpu:
				images = images.cuda()

			optimizer.zero_grad() # zero out grads
			output = model(images) # forward pass

			loss = msecrit(output, images)
			avg_train_loss += loss
			loss.backward() # backward
			optimizer.step() # weight updatepr
		avg_train_loss = avg_train_loss/idx+1

		# TRAINING DONE
		model.eval() # switch to evaluation mode

		n = 0
		avg_loss, avg_ssim, avg_psnr = 0.0, 0.0, 0.0
		with torch.no_grad():
			print("Validation Loop")
			# Testing phase starts
			for idx, (images) in enumerate(tq(iddtestdl)):
				if torch.cuda.is_available() and args.gpu:
					images = images.cuda()
				output = model(images) # forward pass
				loss = msecrit(output, images) # loss calculation
				# calculate the metrics (SSIM and pSNR)
				psnr = psnr_loss(images,output).item()
				avg_loss = ((n * avg_loss) + loss.item()) / (n + 1) # running mean
				avg_psnr = ((n * avg_psnr) + psnr) / (n + 1) # running mean
				n += 1

		avg_psnr = 20.0 * np.log10(avg_psnr) # convert pSNR to dB

		schedular.step()


		logg.append(
			{
				'epoch': epoch,
				'loss': avg_loss,
				'pSNR': avg_psnr,
				# 'SSIM': avg_ssim,
				'lr': optimizer.param_groups[0]['lr']
			}
		) # accumulate information for the .log file

		# name of the log file
		with open(log_file, 'w') as logfile:
			json.dump(logg, logfile)

		# model saving, only if the SSIM is better than before
		if avg_psnr > prev_test_psnr:
			print(colored('[Saving] model saved to {}'.format(model_file), 'red'))
			torch.save({
				'epoch': epoch,
				'model_state': model.state_dict(),
				'optim_state': optimizer.state_dict(),
				'schedular_state': schedular.state_dict(),
				}, os.path.abspath(model_file))
			prev_test_psnr = avg_psnr

		else:
			print(colored('[Saving] model NOT saved'))

if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser(
	"""
	Training script.

	Provide the folder paths which contains traning (patch) images and testing (full-scale mammogram) images produced by dataprep.py.
	Provide a filename prefix to help differentiate model/log from different runs with different configurations.

	(Optionally) You may choose to provide batch size (--batch_size), number of epochs (--epoch), printing interval (--interval).
	(Optionally) You may choose to limit the number of samples (--subset_size) to be used per epoch. '0' means the whole dataset.
	(Optionally) You may also choose to randomize (--randomize_subset) the selected subset every epoch. It has effect only if --subset_size is non-zero.
	(Optionally) You may choose to use GPU (--gpu).
	(Optionally) You may choose the number of Convolution blocks in the architecture. (CVPR paper have 3)
	"""
	)
	parser.add_argument('--traindata', type=str, required=True, help='Folder path of the cropped images (train)')
	parser.add_argument('--testdata', type=str, required=True, help='Folder path to the full-scale mammogram images (test)')
	parser.add_argument('-b', '--batch_size', type=int, required=False, default=32, help='Batch size')
	parser.add_argument('-e', '--epochs', type=int, required=False, default=100, help='Number of epochs to run')
	parser.add_argument('-g', '--gpu', action='store_true', help='Want GPU ?')
	parser.add_argument('-i', '--interval', type=int, required=False, default=50, help='Iteration interval for display')
	parser.add_argument('--model_file_name', type=str, required=True, help='Name of model and log files')
	parser.add_argument('--n_convblocks', type=int, required=False, default=3, help='no of conv blocks in the architecture')
	
	args = parser.parse_args()

	main( args )
