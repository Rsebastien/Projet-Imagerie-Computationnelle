# TODO: fix temporal data augmentation: add scaling
# TODO: temporal validation code not working
# TODO: clean temporal dataset realted functions
# TODO: validation fixed to single sequence ?
"""
Dataset related functions
Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>
This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import glob
import time
import random
import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler
import torchvision.transforms as tr
from models import DVDnet_spatial
from motioncompensation import align_frames
from utils import normalize, open_image, variable_to_cv2_image, \
    augment_train_pair, get_imagenames

DEFAULTMAXNUMPATCHES = 450000  # Default val if max number of patches not set (prepare_data_temp)
NUMSEQS_VAL = 1  # number of sequences to include in validation dataset
NUMFRXSEQ_VAL = 15  # number of frames of each sequence to include in validation dataset
VALSEQPATT = '*'  # pattern for name of validation sequence
TRAINGRAYDBF = 'train_temp_gray.h5'  # names of database files
VALDGRAYBF = 'val_temp_gray.h5'
TRAINRGBDBF = 'train_temp_rgb.h5'
VALRGBDBF = 'val_temp_rgb.h5'


def train_spatial_dataloaders(trainsetdir, patch_size=50, batch_size=128,
                              max_num_patches=1024000, gray_mode=False):
    """Returns the training dataloader.
	"""

    # Input images are assumed to be HxWxC numpy arrays, so	ToPILImage() works well
    tr_choice = tr.RandomChoice([tr.RandomCrop(patch_size),
                                 tr.Compose([tr.RandomCrop(patch_size),
                                             tr.RandomResizedCrop(size=patch_size, scale=(0.9, 0.9), ratio=(1.0, 1.0))
                                             ]),
                                 tr.Compose([tr.RandomCrop(patch_size),
                                             tr.RandomResizedCrop(size=patch_size, scale=(0.8, 0.8), ratio=(1.0, 1.0))
                                             ]),
                                 tr.RandomResizedCrop(size=patch_size, scale=(0.2, 0.2), ratio=(1.0, 1.0)),
                                 tr.Compose([tr.RandomCrop(patch_size),
                                             tr.RandomHorizontalFlip(p=0.5),
                                             tr.RandomVerticalFlip(p=0.5),
                                             tr.ColorJitter(
                                                 brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)])
                                 ])

    transform = tr.Compose([tr.ToPILImage(),
                            tr_choice,
                            tr.ToTensor()])

    dataset_train = SpatialTrainDataset(trainsetdir=trainsetdir,
                                        transform=transform,
                                        gray_mode=gray_mode)

    train_sampler = RandomSampler(data_source=dataset_train,
                                  replacement=True,
                                  num_samples=max_num_patches)

    train_loader = DataLoader(dataset_train,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=9,
                              pin_memory=True,
                              shuffle=False)

    return train_loader


class SpatialTrainDataset(Dataset):
    # TODO: separate function to load images
    """Training dataset. Loads all the images from the dataset on memory. Each time an item is
	requested, the __getitem__(index) method samples the dataset and returns the transformed
	selected image.
	"""

    def __init__(self, trainsetdir=None, transform=None, gray_mode=False):

        self.gray_mode = gray_mode

        # get paths to frames in each dir
        files = get_imagenames(trainsetdir)

        nfiles = len(files)
        if nfiles == 0:
            raise Exception("No images found in the training database")
        else:
            print("Found {} images in the training database".format(len(files)))

        # Append images in dataset to Dataset
        images = []
        for nidx in range(nfiles):
            img = cv2.imread(files[nidx])
            if not gray_mode:
                # HxWxC RGB image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # HxWxC grayscale image (C=1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, -1)
            images.append(img)

        self.transform = transform
        self.images = images

    def __getitem__(self, index):
        return self.transform(self.images[index])

    def __len__(self):
        return len(self.images)


class SpatialValDataset(Dataset):
    # TODO: separate function to load images
    """Validation dataset. Loads all the images in the dataset folder on memory.
	"""

    def __init__(self, valsetdir=None, gray_mode=False):
        self.gray_mode = gray_mode

        # Get ordered list of filenames
        files = get_imagenames(valsetdir)

        nfiles = len(files)
        if nfiles == 0:
            raise Exception("No images found in the training database")
        else:
            print("Found {} images in the validation database".format(len(files)))

        # Append images in dataset to Dataset
        images = []
        for nidx in range(nfiles):
            img = cv2.imread(files[nidx])
            if not gray_mode:
                # HxWxC RGB image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # HxWxC grayscale image (C=1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, -1)
            images.append(img)

        self.images = images

    def __getitem__(self, index):
        return tr.ToTensor()(self.images[index])

    def __len__(self):
        return len(self.images)


def concat_frames(fr_array, noise_map, img_concat):
    r""" Concatenates the [temp_psz, H, W, C] array of the denoised images with the
	CxHxW noise map and the CxHxW central frame.
	Returns:
			 [temp_psz+1+1, C, H, W] float32 in the [0., 1.] range
	"""
    fr_array = normalize(fr_array)  # float32 [0., 1.]
    fr_array = (fr_array).transpose(0, 3, 1, 2)
    img_concat = np.expand_dims(img_concat, axis=0)
    noise_map = np.expand_dims(noise_map, axis=0)

    return np.concatenate((fr_array, noise_map, img_concat), axis=0)


def frames_to_patches(seq, win, stride=1):
    r"""Converts an array of size NxCxHxW composed fo N images to an
		array of 3D patches.
	
	Args:
		seq: a numpy array of sizes NxCxHxW containing N images CxHxW, RGB (C=3)
			or grayscale (C=1) 
		win: spatial size of the output patches
		stride: int. spatial stride
	Returns:
		out: a numpy array of sizes NxCxWINxWINxNUM_PATCHES containing 
			NUM_PATCHES patches of size NxCxWINxWIN
	"""
    assert len(seq.shape) == 4

    k = 0
    endn = seq.shape[0]
    endc = seq.shape[1]
    endw = seq.shape[2]
    endh = seq.shape[3]
    patch = seq[:, :, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    tot_num_patches = patch.shape[-2] * patch.shape[-1]
    out = np.zeros([endn, endc, win * win, tot_num_patches], np.float32)
    for i in range(win):
        for j in range(win):
            patch = seq[:, :, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            out[:, :, k, :] = np.array(patch[:]). \
                reshape(endn, endc, tot_num_patches)
            k = k + 1
    return out.reshape([endn, endc, win, win, tot_num_patches]), tot_num_patches


# TODO: fix this
def prepare_data_temporal(**args):
    r"""Builds the training and validations datasets (RGB) for a denoiser of frame sequences
	by scanning the corresponding directories for images and extracting	patches from them.
	
	Args:
		trainset_dir: path containing the training image dataset
		valset_dir: path containing the validation image dataset
		patchsz: size of the patches to extract from the images
		temp_patchsz: temporal size of the patches to extract (num of contiguous frames)
		stride: size of stride to extract patches
		temp_stride: size of the temporal stride to extract patches
		max_num_patches: maximum number of patches to extract
		aug_times: number of times to augment the available data minus one
		gray_mode: build the databases composed of grayscale patches
		noise_interval: list with noise training interval
		model_spatial_file: path to saved .pth of spatial model
	"""

    assert args['patch_size'] % 2 == 0

    # 	Arguments
    mc_algo = args['motion_comp_algo']
    trainset_dir = args['trainset_dir']
    valset_dir = args['valset_dir']
    patchsz = args['patch_size']
    temp_patchsz = args['temp_patch_size']
    ctrl_fr_idx = int((temp_patchsz - 1) // 2)
    stride = args['stride']
    temp_stride = args['temp_stride']
    max_num_patches = args['max_num_patches']
    noise_interval = list(np.array(args['noise_interval']) / 255.)
    model_file = args['model_spatial_file']

    if args['aug_times'] is None:
        aug_times = 4
    else:
        aug_times = args['aug_times']

    if args['gray_mode'] is None:
        gray_mode = False
    else:
        gray_mode = args['gray_mode']

    # Denoiser model
    if not gray_mode:
        in_ch = 3
    else:
        in_ch = 1

    # net = DVDnet_spatial(rgb_mode=True)
    net = DVDnet_spatial()
    device_ids = [0]
    model_spa = nn.DataParallel(net, device_ids=device_ids).cuda()
    state_dict = torch.load(model_file)
    model_spa.load_state_dict(state_dict)
    model_spa.eval()

    if gray_mode:
        traindbf = TRAINGRAYDBF
        valdbf = VALDGRAYBF
    else:
        traindbf = TRAINRGBDBF
        valdbf = VALRGBDBF

    if max_num_patches is None:
        max_num_patches = DEFAULTMAXNUMPATCHES
        print("\tMaximum number of patches not set, setting value by default")
    print("\tMaximum number of patches set to {}".format(max_num_patches))

    print('> Training database')
    train_num = 0
    t1 = time.time()
    # Look for subdirs with individual sequences
    seqs_dirs = sorted(glob.glob(os.path.join(trainset_dir, '*')))

    with h5py.File(traindbf, 'w') as h5f:

        # save params
        for k in args:
            h5f.attrs[k] = args[k]

        for seq_dir in seqs_dirs:
            # get paths to frames in each dir
            files = get_imagenames(seq_dir)

            fidx = 0  # init temporal index

            # init arrays to handle contiguous frames and related patches
            frstimg, _, _ = open_image(files[0], gray_mode=gray_mode,
                                       expand_if_needed=False)
            _, C, H, W = frstimg.shape
            train_fr_n = torch.zeros((temp_patchsz, C, H, W))
            num_spa_patch = ((W - patchsz) // stride) * ((H - patchsz) // stride)
            train_fr_wrpd = np.zeros((temp_patchsz, H, W, C))
            patchbatch = np.zeros((temp_patchsz, C, patchsz,
                                   patchsz, num_spa_patch))
            train_frames = list()

            while (train_num < max_num_patches) and (fidx + temp_patchsz <= len(files)):

                # if list not yet created, fill it with temp_patchsz frames
                # else, discard first temp_stride frames and refill list
                if not train_frames:
                    for idx in range(temp_patchsz):
                        currimg, _, _ = open_image(files[fidx + idx], gray_mode=gray_mode, expand_if_needed=True,
                                                   expand_axis0=False)
                        train_frames.append(currimg)
                else:
                    for idx in range(temp_stride):
                        del train_frames[0]
                        currimg, _, _ = open_image(files[fidx + 1 + idx],
                                                   gray_mode=gray_mode,
                                                   expand_if_needed=True,
                                                   expand_axis0=False)
                        train_frames.append(currimg)

                # 	add noise and denoise
                # 	std = pick noise from unif distribution
                std_noise = np.random.uniform(noise_interval[0], noise_interval[1])

                # 	add noise to frame array
                train_fr_n = torch.FloatTensor(np.array(train_frames))
                noise = torch.FloatTensor(train_fr_n.shape).normal_(mean=0, std=std_noise)
                train_fr_n = train_fr_n + noise

                with torch.no_grad():
                    train_fr_n = train_fr_n.cuda()
                    # build noise map of size [temp_patchsz, C, H, W]
                    noise_map = torch.full_like(train_fr_n, std_noise).cuda()
                    # 	denoise batch of patch with the spatial denoiser
                    train_fr_den = torch.clamp(model_spa(train_fr_n, noise_map), 0., 1.)

                # convert reference frame to OpenCV img and store it
                # images are BGR
                train_fr_wrpd[ctrl_fr_idx, :, :, :] = variable_to_cv2_image(
                    train_fr_den[ctrl_fr_idx, ...],
                    conv_rgb_to_bgr=False)

                # register frames w.r.t central frame
                # need to convert them to OpenCV images first
                for idx in range(temp_patchsz):
                    if not idx == ctrl_fr_idx:
                        img_to_warp = variable_to_cv2_image(train_fr_den[idx, ...], conv_rgb_to_bgr=False)
                        # train_fr_wrpd is [temp_patchsz, H, W, C]
                        train_fr_wrpd[idx, :, :, :] = align_frames(
                            img_to_warp,
                            train_fr_wrpd[ctrl_fr_idx, ...],
                            mc_alg=mc_algo)

                # concat array with noise map and clean frame
                # train_frames elements are CxHxW
                # train_fr_wrpd is [temp_patchsz, H, W, C]
                noise_map_c = np.full_like(train_frames[ctrl_fr_idx], std_noise)
                # train_pair is an [temp_patchsz+1+1, C, H, W] float32 normalized array
                train_pair = concat_frames(train_fr_wrpd, noise_map_c, train_frames[ctrl_fr_idx])

                for aidx in range(aug_times):
                    train_pair = augment_train_pair(train_pair)

                    # crop patches into a into a [temp_patchsz+1+1, C, Win, Win, Numpatches] array
                    patchbatch, tot_num_realp = frames_to_patches(train_pair,
                                                                  patchsz,
                                                                  stride=stride)

                    # save train pairs [den_patchbatch, noise_map, clean_central_frame]
                    print("\tfile: %s # samples: %d" % (files[fidx], tot_num_realp))
                    for idx in range(tot_num_realp):
                        h5f.create_dataset(str(train_num), \
                                           data=patchbatch[:, :, :, :, idx], \
                                           compression="lzf")
                        train_num += 1
                fidx += temp_stride

    # validation database
    print('\n> Validation database')

    # Look for subdirs with individual sequences
    seqs_dirs = sorted(glob.glob(os.path.join(valset_dir, VALSEQPATT)))
    with h5py.File(valdbf, 'w') as h5f:
        # save params
        for k in args:
            h5f.attrs[k] = args[k]

        for seq_idx, seq_dir in enumerate(seqs_dirs[0:NUMSEQS_VAL], 0):
            print(seq_idx, seq_dir)
            # get paths to frames in each dir
            files = get_imagenames(seq_dir)

            seq_list = list()
            for fpath in files[0:NUMFRXSEQ_VAL]:
                print("\tfile: %s" % fpath)
                img, _, _ = open_image(fpath, gray_mode, expand_if_needed=False, \
                                       expand_axis0=True)
                seq_list.append(img)
            h5f.create_dataset(str(seq_idx), data=np.array(seq_list))

    t2 = time.time()
    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num)
    print('\tvalidation set, # samples %d\n' % (NUMSEQS_VAL * NUMFRXSEQ_VAL))
    print('\telapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))


class DatasetTemp(Dataset):
    r"""Implements torch.utils.data.Dataset
	"""

    def __init__(self, train=True, gray_mode=False, shuffle=False):
        super(DatasetTemp, self).__init__()
        self.train = train
        self.gray_mode = gray_mode
        if self.gray_mode:
            self.traindbf = TRAINGRAYDBF
            self.valdbf = VALDGRAYBF
        else:
            self.traindbf = TRAINRGBDBF
            self.valdbf = VALRGBDBF

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')

        self.keys = list(h5f.keys())
        self.args = dict(h5f.attrs)
        if shuffle:
            random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
