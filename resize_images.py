import numpy as np
from PIL import Image
import os

folder = 'chico_45a'

files = os.listdir('data_irit/chicos/' + folder)

for file in files:
    if file[-4:] == '.jpg' or file[-4:] == '.png':
        number = file[-7:-4]
        print(number)
        img = Image.open('data_irit/chicos/' + folder + '/' + file)
        data = np.asarray(img)
        tmp_shape = [data.shape[0], data.shape[1]]
        if tmp_shape[0] % 2 != 0:
            tmp_shape[0] = data.shape[0] - 1
        if tmp_shape[1] % 2 != 0:
            tmp_shape[1] = data.shape[1] - 1

        new_image = np.zeros((tmp_shape[0], tmp_shape[1]))
        new_image[:, :] = data[:(tmp_shape[0]), :(tmp_shape[1])]

        img2 = Image.fromarray(new_image).convert("L")
        # path = 'data_irit/chicos/' + folder + '/imgs/'
        path = 'data_irit/train_spatial/'
        if not os.path.exists(path):
            os.mkdir(path)
        img2.save(path + folder + '_image-' + number + '.png')