"""
Trains the temporal denoiser of the DVDnet model
By default, the training starts with a learning rate equal to 1e-3 (--lr).
After the number of epochs surpasses the first milestone (--milestone), the
lr gets divided by 100. Up until this point, the orthogonalization technique
described in the DVDnet paper is performed (--no_orthog to set it off).
@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import DVDnet_temporal, DVDnet_spatial
from dataset import DatasetTemp
from utils import svd_orthogonalization, close_logger, init_logging
from train_common import resume_training, lr_scheduler, log_train_temp_psnr, \
    validate_and_log_temporal, save_model_checkpoint


def main(**args):
    r"""Performs the main training loop
    """
    # Load datasets
    print('> Loading dataset ...')
    dataset_train = DatasetTemp(train=True, gray_mode=False, shuffle=True)
    dataset_val = DatasetTemp(train=False, gray_mode=False, shuffle=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=6, \
                              batch_size=args['batch_size'], shuffle=True)

    temp_psz = dataset_train.args['temp_patch_size']  # temporal patch size (how many frames to use)
    assert temp_psz % 2 == 1
    nch = 3  # supported for RGB images
    ctlfr_beg_idx = int((temp_psz - 1) / 2 * nch)  # index of central frame: beginning
    ctlfr_end_idx = int(((temp_psz - 1) / 2 + 1) * nch)  # index of central frame: end

    print("\t# of training samples: %d\n" % int(len(dataset_train)))
    print(len(dataset_val))

    # Init loggers
    writer, logger = init_logging(args)

    # Define GPUs
    device_ids = [0]
    torch.backends.cudnn.benchmark = True

    # Create model
    model_temp = DVDnet_temporal(temp_psz)
    model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()

    # Define loss
    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model_temp.parameters(), lr=args['lr'])

    # Spatial model
    model_spatial = DVDnet_spatial()
    model_spatial = nn.DataParallel(model_spatial, device_ids=device_ids).cuda()
    model_spatial.load_state_dict(torch.load(args['model_spatial_file']))

    # Resume training or start anew
    start_epoch, training_params = resume_training(args, model_temp, optimizer)

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
            model_temp.train()
            model_temp.zero_grad()
            optimizer.zero_grad()

            # extract input frames, noise map, and reference frame from array
            imgs_train = data[:, :temp_psz, ...]
            s0, s1, s2, s3, s4 = imgs_train.size()
            imgs_train = imgs_train.contiguous().view(s0, s1 * s2, s3, s4)
            noise_map = data[:, -2, ...]
            ref_frame = data[:, -1, ...]

            # Send Variables to GPU
            imgs_train = imgs_train.cuda()
            ref_frame = ref_frame.cuda()
            noise_map = noise_map.cuda()

            # Evaluate model and optimize it
            out_train = model_temp(imgs_train, noise_map)
            loss = criterion(out_train, ref_frame) / (imgs_train.size()[0] * 2)
            loss.backward()
            optimizer.step()

            # Log pnsr of results
            if training_params['step'] % args['save_every'] == 0:
                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model_temp.apply(svd_orthogonalization)

            log_train_temp_psnr(torch.clamp(out_train, 0., 1.),
                                ref_frame,
                                imgs_train[:, ctlfr_beg_idx:ctlfr_end_idx, ...],
                                loss,
                                writer,
                                epoch,
                                i,
                                len(loader_train),
                                training_params)

            # The end of each epoch
            training_params['step'] += 1

        # Call to model.eval() to correctly set the BN layers before inference
        model_temp.eval()

        # Validation and log images
        validate_and_log_temporal(model_temp,
                                  model_spatial,
                                  dataset_val,
                                  args['val_noise'] / 255.,
                                  temp_psz,
                                  dataset_train.args['motion_comp_algo'],
                                  writer,
                                  epoch,
                                  current_lr,
                                  logger,
                                  imgs_train)

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model_temp, args, optimizer, training_params, epoch)

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    # Close logger
    close_logger(logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the temporal denoiser of DVDnet")

    # Paths
    parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')
    parser.add_argument("--model_spatial_file", type=str,
                        default="models/rgb/train_paper_20180711_conv12.pth",
                        help='path to model of the pretrained spatial denoiser')

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=20, help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true', \
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Number of training steps to log psnr and perform \
						orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5, \
                        help="Number of training epochs to save state")
    parser.add_argument("--val_noise", type=float, default=25, \
                        help='noise level used on validation set')
    argspar = parser.parse_args()

    print("\n### Training temporal denoiser model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
