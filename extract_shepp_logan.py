from scipy.io import loadmat, savemat
import cv2

mat_filepath = '45a_T2P2_MCT_apexouvert_norm.mat'
voxel_key = 'MCT45a_norm'

if __name__ == '__main__':
    voxelmat_ = loadmat(mat_filepath)
    voxel = voxelmat_[voxel_key]
    for i in range(0, len(voxel)):
        cv2.imwrite("./png_images/frame%(i)d.jpg" % {'i': i}, (voxel[i] * 255))

    png_filepath = mat_filepath.split('.')[0] + '.png'
