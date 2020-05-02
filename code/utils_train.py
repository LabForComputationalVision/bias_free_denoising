import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as plt
from math import isnan, sqrt
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from torch.optim import Adam
from skimage import io
# from sklearn.metrics import mean_squared_error
from PIL import Image
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim
from math import sqrt
################################################# Helper Functions #################################################




# rewrite the load function into more general one
def load_Berkley_dataset( train_folder_path, test_folder_path,set12_path):
    image_dict = {};
    # read and prep train images
    train_images = []
    train_names = os.listdir(train_folder_path)
    test_names = os.listdir(test_folder_path)

    for file_name in train_names:
        train_images.append(io.imread(train_folder_path + file_name).astype(float)/255 );

    # read and prep test images
    test_images = []
    for file_name in test_names:
        image = io.imread(test_folder_path + file_name).astype(float)
        if image.shape[0] > image.shape[1]:
            image = image.T
        test_images.append(image/255 );

    #read and prep set12
    images_set12 = []
    set12_names = os.listdir(set12_path)

    for file_name in set12_names:
        images_set12.append(io.imread(set12_path + file_name).astype(float)/255 );

    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)
    image_dict['set12'] = np.array(images_set12)

    return image_dict

def single_image_loader(data_set_dire_path, image_number):
    all_names = os.listdir(data_set_dire_path)
    file_name = all_names[image_number]
    im = io.imread(data_set_dire_path + file_name).astype(float)/255
    return im


def add_noise(all_patches, noise_level, mode='B'):
    all_patches_noisy = []
    all_noises = []

    for i in range(all_patches.shape[0]):
        #for blind denoising
        if mode == 'B':
            std = np.int(np.random.randint(noise_level[0], noise_level[1] , size = 1))/255
        #for specific noise
        else:
            std = noise_level/255
        noise =np.random.normal(loc=0.0, scale=  std, size= (all_patches.shape[1], all_patches.shape[2]))
        all_patches_noisy.append( all_patches[i:i+1] + noise)
        all_noises.append(noise)

    return np.concatenate(all_patches_noisy, axis = 0), np.stack(all_noises, axis = 0)






def patch_generator_with_scale(all_images, patch_size, stride, scales, resample = Image.BICUBIC):
    '''images: a 3D numpy array of image
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    scales: a list of float values by which the image is scaled
    '''

    all_images_patches = []
    # loop through all images in the set
    for image in all_images:
        image_patches = [] # holder for all patches of one image from different scales
        # loop through all the scales in the list
        for i in range(len(scales)):
            # resize the image (and blur if needed)
            image_pil = Image.fromarray(image)
            # if blur is True:
                 # image_pil = image_pil.convert('L').filter(ImageFilter.GaussianBlur(1))
                 # image_pil = image_pil.convert('F')
            newsize = (int(image_pil.size[0] * scales[i]), int(image_pil.size[1] * scales[i]))
            image_pil_resize = image_pil.resize(newsize, resample=resample)
            image_re = np.array(image_pil_resize)


            im_height = image_re.shape[0]
            im_width = image_re.shape[1]


            patches = []
            h = int(im_height/stride[0]) * stride[0]
            w = int(im_width/stride[1]) * stride[1]
            # create patches for an image of a certain scale
            for x in range(0,h- patch_size[0] + 1, stride[0]):
                for y in range(0,w - patch_size[1] + 1, stride[1]):
                    # patches[counter] = image_re[ x:x+patch_size[0] , y:y+patch_size[1]]
                    patch = image_re[ x:x+patch_size[0] , y:y+patch_size[1]]
                    # patches.append(patch.reshape(1, patch.shape[0], patch.shape[1])) # add a dimension
                    patches.append(patch) # add a dimension

            patches = np.stack(patches, 0) # all the patches from one image at one scale
            image_patches.append(patches)
        image_patches = np.concatenate(image_patches, axis=0)
        all_images_patches.append(image_patches)
    return np.concatenate(all_images_patches, axis=0)


def plot_loss(loss_list, loss_list_test, loss_list_ave, dir_name,epochs):

    plt.figure(figsize= (12,8))
    plt.plot( np.linspace(0,len(loss_list),len(loss_list_test) ), np.log(loss_list_test),'red', label = 'test loss')
    plt.plot(np.log(loss_list), 'blue',label = 'train loss', alpha = .5)
    plt.ylabel('log_e(loss)')
    plt.title('final average test loss ' + str(loss_list_test[-1]))
    plt.legend()
    plt.savefig(dir_name  + '/loss.png')

    plt.figure(figsize= (12,8))
    plt.plot( range(len(loss_list_test)), loss_list_test, 'r-o', label = 'test loss')
    plt.plot( range(len(loss_list_ave)), loss_list_ave, 'b-o', label = 'ave train loss')
    plt.ylabel('loss')
    plt.title('minimum average test loss ' + str(min(loss_list_test)) + ' from epoch ' + str(loss_list_test.index(min(loss_list_test))))
    plt.legend()
    plt.savefig(dir_name  + '/loss_non_log.png')


def plot_denoised_image(all_params, images_clean, images_noisy, residuals ):
    noise_level_range = all_params['noise_level_range']
    parent_included = all_params['parent_included']
    dir_name = all_params['dir_name']
    kernel_size_waves = all_params['kernel_size_waves']
    kernel_size_mid = all_params['kernel_size_mid']

    n = images_clean.shape[0]

    residuals_np = residuals.data.squeeze(1).cpu().numpy()

    f, axes = plt.subplots(n, 3)
    f.suptitle('noise level: random' + ',denoising kernel size: ' + str(kernel_size_mid) + ',decomp. kernel size: ' + str(kernel_size_waves) + ', parents included: ' + str(parent_included) , fontsize= 6)
    ax = axes.ravel()

    for i ,j in zip(range(0,n*3, 3), range(n) ):
        image_c = images_clean[j]
        ax[i].imshow(image_c, 'gray', vmin=0, vmax = 1)
        ax[i].set_axis_off()

        images_n = np.clip(images_noisy[j] , 0,1)
        ax[i+1].imshow(images_n, 'gray', vmin=0, vmax = 1)
        ax[i+1].set_title( 'PSNR '+ str(compare_psnr(images_n, image_c, data_range=1)) + '\n SSIM ' + str(compare_ssim(image_c, images_n )), fontsize = 5)
        ax[i+1].set_axis_off()

        image_d = np.clip((images_noisy[j] - residuals_np[j]) , 0,1)
        ax[i+2].imshow(image_d, 'gray', vmin=0, vmax = 1)
        ax[i+2].set_title(  'PSNR '+ str(compare_psnr(image_d,image_c, data_range=1)) + '\n SSIM ' + str(compare_ssim(image_c,image_d )),fontsize = 5)
        ax[i+2].set_axis_off()

    file_name = dir_name + '/denoised_test_image.png'
    plt.savefig(file_name, dpi = 500)
    plt.close('all')


def plot_set12(all_params, image_clean, image_noisy, residual ,h):
    noise_level_range = all_params['noise_level_range']
    parent_included = all_params['parent_included']
    dir_name = all_params['dir_name']
    kernel_size_waves = all_params['kernel_size_waves']
    kernel_size_mid = all_params['kernel_size_mid']

    residual_np  = residual.data.squeeze(0).squeeze(0).cpu().numpy()

    f, axes = plt.subplots(1, 3, figsize= (12,4))
    f.suptitle('noise level: '+ str(noise_level_range)+ ',denoising kernel size: ' + str(kernel_size_mid) + ',decomp. kernel size: ' + str(kernel_size_waves) + ', parents included: ' + str(parent_included) , fontsize= 6)
    ax = axes.ravel()

    image_c = image_clean
    ax[0].imshow(image_c, 'gray')
    ax[0].set_axis_off()

    image_n = np.clip(image_noisy , 0,1)
    ax[1].imshow(image_n, 'gray')
    ax[1].set_title( 'PSNR '+ str(compare_psnr(image_n, image_c, data_range = 1)) + '\n SSIM ' + str(compare_ssim(image_c, image_n )), fontsize = 5)
    ax[1].set_axis_off()

    image_d = np.clip((image_noisy - residual_np), 0,1)
    ax[2].imshow(image_d, 'gray')
    ax[2].set_title(  'PSNR '+ str(compare_psnr(image_d, image_c, data_range=1)) + '\n SSIM ' + str(compare_ssim(image_c,image_d )),fontsize = 5)
    ax[2].set_axis_off()

    file_name = dir_name + '/denoised_set12_'+str(h)+'.png'

    plt.savefig(file_name, dpi = 500)
    plt.close('all')


def data_augmentation(image,mode): # reference: https://github.com/SaoYan/DnCNN-PyTorch

    if mode == 1:
        return image

    if mode == 2: # flipped
        image = np.flipud(image);
        return image

    elif mode == 3: # rotation 90
        image = np.rot90(image,1);
        return image;

    elif mode == 4 :# rotation 90 & flipped
        image = np.rot90(image,1);
        image = np.flipud(image);
        return image;

    elif mode == 5: # rotation 180
        image = np.rot90(image,2);
        return image;

    elif mode == 6: # rotation 180 & flipped
        image = np.rot90(image,2);
        image = np.flipud(image);
        return image;

    elif mode == 7: # rotation 270
        image = np.rot90(image,3);
        return image;

    elif mode == 8: # rotation 270 & flipped
        image = np.rot90(image,3);
        image = np.flipud(image);
        return image;
    else:
        raise ValueError('the requested mode is not defined')


def augment_training_data(train_set): # reference: https://github.com/SaoYan/DnCNN-PyTorch

    augmented_train_set = np.zeros_like(train_set)
    for i in range(train_set.shape[0]):
        mode = np.random.randint(1,9)
        augmented_train_set[i,:,:] =  data_augmentation(train_set[i,:,:], mode)

    train_set = np.concatenate((train_set, augmented_train_set))
    return train_set

def weights_init_kaiming(m): # reference: https://github.com/SaoYan/DnCNN-PyTorch
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    #elif classname.find('BatchNorm') != -1:
    #    m.weight.data.normal_(mean=0, std=sqrt(2./9./64.)).clamp_(-0.025,0.025)
    #    nn.init.constant(m.bias.data, 0.0)
