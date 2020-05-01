import numpy as np
from math import isnan, sqrt
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from torch.optim import Adam
from skimage import io
from sklearn.metrics import mean_squared_error
from utils_train import *
from DnCNN import *
from BF_DnCNN import *
from skimage.measure.simple_metrics import compare_psnr, compare_mse


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
    for cnn in [ 'BF_DnCNN', 'DnCNN' ]:
        folder_path = '../models/'+cnn+'/range_'+str(l)+'_'+str(h)+'/'

        if cnn == 'BF_DnCNN':
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

        elif cnn == 'DnCNN':
            model = nn.DataParallel(dncnn(all_params))
            if torch.cuda.is_available():
                model = model.cuda()
                model.load_state_dict(torch.load(folder_path + 'model.pt'))
            else:
                model.load_state_dict(torch.load(folder_path +'model.pt', map_location='cpu' ))

        else:
            raise Exception("model name not valid")

        model.eval()

        psnr68_list = []
        for noise in range(0,110, 10):
            print('calculating loss for noise  : ', noise)
            noisy68, noise68 = add_noise(all_images['test'], noise, 'S')
            if torch.cuda.is_available():
                inp68 = Variable(torch.FloatTensor(noisy68).unsqueeze(1).cuda(), volatile = True,requires_grad=False)
            else:
                inp68 = Variable(torch.FloatTensor(noisy68).unsqueeze(1), volatile = True,requires_grad=False)

            psnr68 = 0
            for i in range(all_images['test'].shape[0]):
                output68_ind = model(inp68[i:i+1]) #gives the residual
                denoised68_ind =inp68[i:i+1] - output68_ind
                denoised68_ind_np = np.clip((denoised68_ind.data.squeeze(0).squeeze(0).cpu().numpy()) , 0,1)
                psnr68 +=  compare_psnr(denoised68_ind_np.astype('float32'), all_images['test'][i].astype('float32')  ,data_range = 1)
            psnr68_list.append(psnr68/all_images['test'].shape[0])



        if not os.path.exists(folder_path  + 'results/' ):
            os.makedirs(folder_path  + 'results/')

        np.save(folder_path + 'results/psnr68_list.npy', psnr68_list)

if __name__ == "__main__" :
    main()
