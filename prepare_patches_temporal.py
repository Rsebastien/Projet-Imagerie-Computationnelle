"""
Construction of the training and validation databases
@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import argparse
from dataset import prepare_data_temporal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building the training \
													patch database")
    # Preprocessing parameters
    parser.add_argument("--motion_comp_algo", "--mc", type=str, default="DeepFlow", \
                        choices=["SimpleFlow", "DeepFlow", "TVL1"], \
                        help="prepare grayscale database instead of RGB")
    parser.add_argument("--gray_mode", action='store_true', \
                        help="prepare grayscale database instead of RGB")
    parser.add_argument("--patch_size", "--p", type=int, default=44, \
                        help="Patch size")
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, \
                        help="Temporal patch size, i.e. number of contiguous frames")
    parser.add_argument("--stride", "--s", type=int, default=20, help="Size of stride")
    parser.add_argument("--temp_stride", "--ts", type=int, default=3,
                        help="Temporal stride")
    parser.add_argument("--max_num_patches", "--m", type=int, default=450000,
                        help="Maximum number of patches")
    parser.add_argument("--aug_times", "--a", type=int, default=1, \
                        help="How many times to perform data augmentation")
    parser.add_argument("--noise_interval", nargs=2, type=int, \
                        default=[0, 55], help="Noise training interval")
    # Dirs
    parser.add_argument("--trainset_dir", type=str, default=None, \
                        help='path of trainset')
    parser.add_argument("--valset_dir", type=str, default=None, \
                        help='path of validation set')
    parser.add_argument("--model_spatial_file", type=str, default=None, \
                        help='path denoiser model')
    args = parser.parse_args()

    print("\n### Building databases ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    prepare_data_temporal(**vars(args))