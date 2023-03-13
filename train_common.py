"""
Different common functions for training the spatial and temporal DVDnet models.
Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>
This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import time
import torch
import torch.nn as nn
import torchvision.utils as tutils
from utils import batch_psnr
from dvdnet import denoise_seq_dvdnet


def resume_training(argdict, model, optimizer):
    """ Resumes previous training or starts anew
	"""
    if argdict['resume_training']:
        resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = argdict['epochs']
            new_milestone = argdict['milestone']
            current_lr = argdict['lr']
            argdict = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            argdict['epochs'] = new_epoch
            argdict['milestone'] = new_milestone
            argdict['lr'] = current_lr
            print("=> loaded checkpoint '{}' (epoch {})" \
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = checkpoint['args']
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            argdict['resume_training'] = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}". \
                            format(resumef))
    else:
        start_epoch = 0
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = argdict['no_orthog']

    return start_epoch, training_params


def lr_scheduler(epoch, argdict):
    """Returns the learning rate value depending on the actual epoch number
	"""
    # Learning rate value scheduling according to argdict['milestone']
    reset_orthog = False
    if epoch > argdict['milestone'][1]:
        current_lr = argdict['lr'] / 1000.
        reset_orthog = True
    elif epoch > argdict['milestone'][0]:
        current_lr = argdict['lr'] / 10.
    else:
        current_lr = argdict['lr']
    return current_lr, reset_orthog


def log_train_spatial_psnr(result, imsource, loss, writer, epoch, idx, num_minibatches, training_params):
    # Compute pnsr of the whole batch
    psnr_train = batch_psnr(torch.clamp(result, 0., 1.), imsource, 1.)

    # Log the scalar values
    writer.add_scalar('loss', loss.item(), training_params['step'])
    writer.add_scalar('PSNR on training data', psnr_train, \
                      training_params['step'])
    print("[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f}". \
          format(epoch + 1, idx + 1, num_minibatches, loss.item(), psnr_train))


def validate_and_log_spatial(model, dataset_val, valnoisestd, writer, epoch, lr, logger, trainimg):
    """Validation step after the epoch finished
	"""
    psnr_val = 0
    with torch.no_grad():
        for valimg in dataset_val:
            img_val = torch.unsqueeze(valimg, 0)
            noise = torch.FloatTensor(img_val.size()). \
                normal_(mean=0, std=valnoisestd)
            imgn_val = img_val + noise
            img_val = img_val.cuda()
            imgn_val = imgn_val.cuda()
            noise_map_val = torch.full_like(imgn_val, valnoisestd).cuda()
            out_val = torch.clamp(model(imgn_val, noise_map_val), 0., 1.)
            psnr_val += batch_psnr(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', lr, epoch)

        # Log val images
        try:
            idx = 0
            if epoch == 0:
                # Log graph of the model
                writer.add_graph(model, (imgn_val, noise_map_val), )
                # Log validation images
                imclean = tutils.make_grid(img_val.data[idx].clamp(0., 1.),
                                           nrow=2, normalize=False, scale_each=False)
                imnsy = tutils.make_grid(imgn_val.data[idx].clamp(0., 1.),
                                         nrow=2, normalize=False, scale_each=False)
                writer.add_image('Clean validation image {}'.format(idx), imclean, epoch)
                writer.add_image('Noisy validation image {}'.format(idx), imnsy, epoch)

            # log output
            imrecons = tutils.make_grid(out_val.data[idx].clamp(0., 1.),
										nrow=2, normalize=False, scale_each=False)
            writer.add_image('Reconstructed validation image {}'.format(idx), imrecons, epoch)

            # Log training images
            imclean = tutils.make_grid(trainimg.data, nrow=8, normalize=True,
                                       scale_each=True)
            writer.add_image('Training patches', imclean, epoch)

        except Exception as e:
            logger.error("validate_and_log_spatial(): Couldn't log results, {}".format(e))


def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
    """Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
	Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
	"""
    torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
    save_dict = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'training_params': train_pars,
                 'args': argdict}
    torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))

    if epoch % argdict['save_every_epochs'] == 0:
        torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt_e{}.pth'.format(epoch + 1)))
    del save_dict


def log_train_temp_psnr(result, imref, ctrlfr, loss, writer, epoch, idx, num_minibatches, training_params):
    """Computes and logs the PSNR of the training result with respect to the reference frame,
	and w.r.t. the spatially denoised central frame
	"""
    psnr_train = batch_psnr(result, imref, 1.)
    # TODO: is this really the central frame or the view() above shuffled the channels?
    psnr_ctrlfr = batch_psnr(result, ctrlfr, 1.)

    # Log the scalar values
    writer.add_scalar('loss', loss.item(), training_params['step'])
    writer.add_scalar('PSNR on training data', psnr_train, training_params['step'])
    writer.add_scalar('PSNR of central frame', psnr_ctrlfr, training_params['step'])
    print("[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f}". \
          format(epoch + 1, idx + 1, num_minibatches, loss.item(), psnr_train))


def validate_and_log_temporal(model_temp, model_spatial, dataset_val, \
                              valnoisestd, temp_psz, mc_algo, writer, epoch, lr, logger, trainimg):
    """Validation step after the epoch finished
	"""
    t1 = time.time()
    psnr_val = 0
    with torch.no_grad():
        for seq_val in dataset_val:
            noise = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
            seqn_val = seq_val + noise
            seq_val = seq_val.cuda()
            seqn_val = seqn_val.cuda()
            sigma_noise = torch.cuda.FloatTensor([valnoisestd])
            out_val = denoise_seq_dvdnet(seq=seqn_val,
                                         noise_std=sigma_noise,
                                         temp_psz=temp_psz,
                                         model_temporal=model_temp,
                                         model_spatial=model_spatial,
                                         mc_algo=mc_algo)
            psnr_val += batch_psnr(out_val, seq_val.squeeze_(), 1.)
        psnr_val /= len(dataset_val)
        t2 = time.time()
        print("\n[epoch %d] PSNR_val: %.4f, on %.2f sec" % (epoch + 1, psnr_val, (t2 - t1)))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', lr, epoch)

    # Log val images
    try:
        idx = 0
        if epoch == 0:
            # Log graph of the model
            # 			_, _, C, H, W = seqn_val.size()
            # 			writer.add_graph(model_temp, (seqn_val, sigma_noise.expand((1, C, H, W)), ), verbose=True)

            # Log validation images
            img = tutils.make_grid(seq_val.data[idx].clamp(0., 1.),
                                   nrow=2, normalize=False, scale_each=False)
            imgn = tutils.make_grid(seqn_val.data[idx].clamp(0., 1.),
                                    nrow=2, normalize=False, scale_each=False)
            writer.add_image('Clean validation image {}'.format(idx), img, epoch)
            writer.add_image('Noisy validation image {}'.format(idx), imgn, epoch)

        # Log validation results
        irecon = tutils.make_grid(out_val.data[idx].clamp(0., 1.),
                                  nrow=2, normalize=False, scale_each=False)
        writer.add_image('Reconstructed validation image {}'.format(idx), irecon, epoch)

        # Log training images
        _, _, Ht, Wt = trainimg.size()
        img = tutils.make_grid(trainimg.data[idx].view(-1, 3, Ht, Wt), \
                               nrow=8, normalize=True, scale_each=True)
        writer.add_image('Training patches', img, epoch)

    except Exception as e:
        logger.error("validate_and_log_temporal(): Couldn't log results, {}".format(e))
