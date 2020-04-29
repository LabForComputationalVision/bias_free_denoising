import numpy as np
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from skimage import io
from DnCNN import *
from utils_analysis import *



def main():
    train_folder_path = '../data/Train400/'
    test_folder_path = '../data/Test/Set68/'
    set12_path = '../data/Test/Set12/'

    all_images= load_Berkley_dataset( train_folder_path, test_folder_path, set12_path)

    all_params = {
    "kernel_size_mid" : 3,
    "padding_mid" : 1,
    "num_mid_kernels" : 64,
    "num_mid_layers" : 20,
    'im_d': 20 #patch dimension
    }

    l = 0  # lower bound of training range
    h = 10 # higher bound of training range

    ############################## load the  model ##############################
    folder_path = '../models/DnCNN/range_'+str(l)+'_'+str(h)+'/'

    model = nn.DataParallel(dncnn(all_params))
    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(folder_path + 'model.pt'))
    else:
        model.load_state_dict(torch.load(folder_path +'model.pt', map_location='cpu' ))


    model.eval()

    ##############################  clean image prep ##############################
    im_d = all_params['im_d'] ## here choose the size of the patch

    I = np.eye(int(im_d*im_d))

    norm_res_ave = []
    norm_b_ave = []
    norm_noise_ave = []


    norm_res = {}
    norm_b = {}
    norm_noise = {}


    if not os.path.exists(folder_path  + 'results/' ):
        os.makedirs(folder_path  + 'results/')

    for noise_level in range(5,105,5):
        print('noise level -------- ' , noise_level)
        norm_res[noise_level]  = []
        norm_b[noise_level]  = []
        norm_noise[noise_level]  = []


        for exp in range(100):
            # random selection of a patch
            image_num = np.random.randint(0,all_images['test'].shape[0])
            im_c = all_images['test'][image_num:image_num+1]
            coor1  = np.random.randint(0,all_images['test'].shape[1]-im_d)
            coor2  = np.random.randint(0,all_images['test'].shape[2]-im_d)
            im_clean = im_c[0, coor1:coor1+im_d, coor2:coor2+im_d]

            im_clean = im_clean.reshape(1,im_d,im_d)
            im , noise = add_noise(im_clean, [noise_level,noise_level+1], 'B')

            im = im[0]

            if torch.cuda.is_available():
               inp = Variable(torch.FloatTensor(im ).unsqueeze(0).unsqueeze(0).cuda(),requires_grad=True)
            else:
               inp = Variable(torch.FloatTensor(im ).unsqueeze(0).unsqueeze(0),requires_grad=True)

            res = model(inp)
            im_denoised = im - res.cpu().data.squeeze(0).squeeze(0).numpy()

            J_im = calc_jacobian(im,model,all_params)
            My = np.dot(J_im.cpu().numpy(), im.flatten()).reshape(im_d,im_d)
            b_im = My - res.cpu().data.squeeze(0).squeeze(0).numpy()
            norm_b[noise_level].append( np.linalg.norm(b_im))
            norm_noise[noise_level].append( np.linalg.norm(noise))
            norm_res[noise_level].append( np.linalg.norm(im - im_denoised))

        # Take averages
        norm_b_ave.append(np.mean(norm_b[noise_level]))
        norm_noise_ave.append(np.mean(norm_noise[noise_level]))
        norm_res_ave.append(np.mean(norm_res[noise_level]))

    np.save(folder_path + 'results'  + '/norm_b_ave.npy',  norm_b_ave)
    np.save(folder_path + 'results'  + '/norm_res_ave.npy',  norm_res_ave)
    np.save(folder_path + 'results'  + '/norm_noise_ave.npy',  norm_noise_ave)


if __name__ == "__main__" :
    main()
