"""
Trains a the spatial denoiser of the DVDnet model

By default, the training starts with a learning rate equal to 1e-3 (--lr).
After the number of epochs surpasses the first milestone (--milestone), the
lr gets divided by 100. Up until this point, the orthogonalization technique
is performed (--no_orthog to set it off).

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import DVDnet_spatial
from dataset import train_spatial_dataloaders, SpatialValDataset
from utils import svd_orthogonalization, close_logger, init_logging
from train_common import resume_training, lr_scheduler, log_train_spatial_psnr, \
    validate_and_log_spatial, save_model_checkpoint


def main(**args):
    r"""Performs the main training loop
	"""
    # Load dataset
    print('> Loading datasets ...')
    dataset_val = SpatialValDataset(valsetdir=args['valset_dir'], gray_mode=False)
    loader_train = train_spatial_dataloaders(trainsetdir=args['trainset_dir'], \
                                             patch_size=args['patch_size'], \
                                             batch_size=args['batch_size'], \
                                             max_num_patches=args['max_number_patches'], \
                                             gray_mode=False)

    num_minibatches = int(args['max_number_patches'] // args['batch_size'])
    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    # Init loggers
    writer, logger = init_logging(args)

    # Define GPU devices
    device_ids = [0]
    torch.backends.cudnn.benchmark = True

    # Create model
    model = DVDnet_spatial()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # Define loss
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # Resume training or start anew
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Training
    start_time = time.time()
    for epoch in range(start_epoch, args['epochs']):
        # Set learning rate
        current_lr, reset_orthog = lr_scheduler(epoch, args)
        if reset_orthog:
            training_params['no_orthog'] = True

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):
            # Pre-training step
            model.train()

            # When optimizer = optim.Optimizer(model.parameters()) we only zero the optim's grads
            optimizer.zero_grad()

            # inputs: noise and noisy image
            # TODO: improve code using torch operations
            img_train = data
            noise = torch.zeros(img_train.size())
            stdn = np.random.uniform(args['noise_ival'][0], args['noise_ival'][1], \
                                     size=noise.size()[0]).astype('float32')
            for nx in range(noise.size()[0]):
                sizen = noise[0, :, :, :].size()
                noise[nx, :, :, :] = torch.FloatTensor(sizen). \
                    normal_(mean=0, std=stdn[nx].astype('float'))
            imgn_train = img_train + noise

            # Send tensors to GPU
            img_train = img_train.cuda()
            imgn_train = imgn_train.cuda()
            noise = noise.cuda()
            noise_map = torch.tensor(stdn.reshape(-1, 1, 1, 1)). \
                expand(*imgn_train.size()).cuda()

            # Evaluate model and optimize it
            out_train = model(imgn_train, noise_map)
            # Training with residual model, so loss must be calculated w.r.t. noise
            loss = criterion(img_train, out_train) / (imgn_train.size()[0] * 2)
            loss.backward()
            optimizer.step()

            # Results
            if training_params['step'] % args['save_every'] == 0:
                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Compute training PSNR
                log_train_spatial_psnr(out_train,
                                       img_train,
                                       loss,
                                       writer,
                                       epoch,
                                       i,
                                       num_minibatches,
                                       training_params)

            # The end of each epoch
            training_params['step'] += 1

        # Call to model.eval() to correctly set the BN layers before inference
        model.eval()

        # Validation and log images

        validate_and_log_spatial(model,
                                 dataset_val,
                                 args['val_noiseL'],
                                 writer,
                                 epoch,
                                 current_lr,
                                 logger,
                                 img_train)

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    # Close logger file
    close_logger(logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the spatial denoiser of DVDnet")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=20, \
                        help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true', \
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3, \
                        help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true', \
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10, help="Number of training steps to log psnr and perform \
						orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5, \
                        help="Number of training epochs to save state")
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[0, 55], \
                        help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=25, \
                        help='noise level used on validation set')
    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=50, \
                        help="Patch size")
    parser.add_argument("--max_number_patches", "--m", type=int, default=102400, \
                        help="Maximum number of patches")
    # Dirs
    parser.add_argument("--log_dir", type=str, default="logs", \
                        help='path of log files')
    parser.add_argument("--trainset_dir", type=str, default=None, \
                        help='path of trainset')
    parser.add_argument("--valset_dir", type=str, default=None, \
                        help='path of validation set')
    argspar = parser.parse_args()

    if argspar.trainset_dir is None:
        argspar.trainset_dir = 'data/rgb/CImageNet_expl'
    if argspar.valset_dir is None:
        argspar.valset_dir = 'data/rgb/Kodak24'

    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training spatial denoiser model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
