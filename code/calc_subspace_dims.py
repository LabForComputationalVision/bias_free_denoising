import numpy as np
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from skimage import io
from BF_DnCNN import *
from utils_analysis import *

def main():
    train_folder_path = '../data/Train400/'
    test_folder_path = '../data/Test/Set68/'
    set12_path = '../data/Test/Set12/'

    all_images= load_Berkley_dataset( train_folder_path, test_folder_path, set12_path)
    all_params = {
    'epochs': 50,
    "learning_rate" : .001,
    'parent_included': 'not applicable',
    'kernel_size_waves': 'not applicable',

    "kernel_size_mid" : 3,
    "padding_mid" : 1,
    "num_mid_kernels" : 64,
    "num_mid_layers" : 20,

    "batch_size" : 128,
    "patch_size" : (50,50),
    "patching_strides" : (10,10),
    'scales' : [1,.9,.8,.7],
    }

    # select the noise range models have been trained on.
    l = 0  # lower bound of training range
    h = 10 # higher bound of training range

    # If you have saved your models with different names, change the list of cnn names below.
    folder_path = '../models/BF_DnCNN/range_'+str(l)+'_'+str(h)+'/'

    model = bf_dncnn(all_params)
    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(folder_path+'model.pt'))
        running_sd = torch.load(folder_path+'running_sd.pt')
        for i in range(all_params['num_mid_layers']-2):
            model.running_sd[i] = running_sd[i]
    else:
        model.load_state_dict(torch.load(folder_path + 'model.pt', map_location='cpu' ))
        running_sd = torch.load(folder_path +'running_sd.pt', map_location='cpu' )
        for i in range(all_params['num_mid_layers']-2):
            model.running_sd[i] = running_sd[i]


    model.eval()


    dims = {}

    ############################## across images ##############################
    for exp in range(10):
        print('image # ', exp)
        dims[exp] = []

        image_num = np.random.randint(0,all_images['test'].shape[0])
        im_c = all_images['test'][image_num:image_num+1]
        coor1  = np.random.randint(0,all_images['test'].shape[1]-im_d)
        coor2  = np.random.randint(0,all_images['test'].shape[2]-im_d)
        im_clean = im_c[0, coor1:coor1+im_d, coor2:coor2+im_d]
        im_clean = im_clean.reshape(1,im_d,im_d)

        init_noise_level =  [10,11] ## here choose the test noise std you want to add to the image

        _ , init_noise = add_noise(im_clean, init_noise_level, 'B')

        ############################## across noise levels ##############################
        for i in range(1,11):

            test_noise = init_noise*i
            print('noise level ', np.std(test_noise))
            im = im_clean + test_noise
            im = im[0]


            J_im = calc_jacobian(im,model,all_params)
            U, S, V = np.linalg.svd(np.eye(n,n) - J_im.cpu().numpy())
            dims[exp].append(int(np.sum((S)**2)))






    np.save(all_params['dir_name'] +'/' +all_params['folder_name']  + '/dims_many_im.npy', dims)


if __name__ == "__main__" :
    main()
