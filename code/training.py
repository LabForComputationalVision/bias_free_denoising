import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
# from math import isnan, sqrt
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from torch.optim import Adam
from skimage import io
# from sklearn.metrics import mean_squared_error
from skimage.measure.simple_metrics import compare_psnr
from torch.utils.data import DataLoader
from helper_functions import *
from network import *


################################################# training #################################################
def train_entire_net(all_images,  all_params):
    start_time_total = time.time()

    epochs = all_params['epochs']
    noise_level_range = all_params['noise_level_range']
    learning_rate = all_params['learning_rate']
    batch_size = all_params['batch_size']
    dir_name = all_params['dir_name']
    patching_strides = all_params['patching_strides']
    scales = all_params['scales']

    num_mid_layers = all_params['num_mid_layers']
    kernel_size_mid = all_params['kernel_size_mid']
    padding_mid = all_params['padding_mid']
    num_mid_kernels= all_params['num_mid_kernels']
    patch_size = all_params['patch_size']

    train_images = patch_generator_with_scale(all_images['train'],  patch_size, patching_strides , scales )
    train_images = augment_training_data(train_images) ## double check data augmenttation repeats
    N = train_images.shape[0]
    print('number of patches ', N)
    trainloader = DataLoader(dataset=train_images, batch_size=batch_size, shuffle=True)

    test_images_noisy, test_noises = add_noise(all_images['test'], noise_level_range, 'B') # used to calculate test loss

    # Prepare test Variables
    if torch.cuda.is_available():
        inp_test = Variable(torch.FloatTensor(test_images_noisy).unsqueeze(1).cuda(), volatile = True,requires_grad=False)
        test_noises = Variable(torch.FloatTensor(test_noises).unsqueeze(1).cuda(), volatile = True,requires_grad=False)
    else:
        inp_test = Variable(torch.FloatTensor(test_images_noisy).unsqueeze(1), volatile = True,requires_grad=False)
        test_noises = Variable(torch.FloatTensor(test_noises).unsqueeze(1), volatile = True,requires_grad=False)

    model = Net(all_params)
    model.apply(weights_init_kaiming) #from pytorch implementation . Not sure what it deos

    if torch.cuda.is_available():
        print('[ Using CUDA ]')
          #torch.cuda.device(0)
        #model = nn.DataParallel(model).cuda()
        model = model.cuda()

    print('number of parameters is ' , sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.MSELoss(size_average=False) # If the field size_average is set to False, the losses are instead summed for each minibatch.
    # optimizer = Adam(model.parameters(), lr = learning_rate)
    optimizer = Adam(filter(lambda p: p.requires_grad,model.parameters()), lr = learning_rate)

    loss_list = []
    loss_list_ave = []
    loss_list_test = []

    # loop over epochs
    for h in range(epochs):
        counter = 0
        loss_sum = 0
        print('epoch ', h )
        if h >= 50 and h%10==0:
            for param_group in optimizer.param_groups: # changed this
                learning_rate = learning_rate/2
                param_group["lr"] = learning_rate
        #loop over images
        for i, batch in enumerate(trainloader, 0):
            counter += 1
            model.train() # added this
            optimizer.zero_grad()
            noisy_batch, noises = add_noise(batch.numpy(), noise_level_range, 'B')
            # Prepare Variables
            if torch.cuda.is_available():
                inp = Variable(torch.FloatTensor(noisy_batch).unsqueeze(1).cuda())
                noises = Variable(torch.FloatTensor(noises).unsqueeze(1).cuda())
            else:
                inp = Variable(torch.FloatTensor(noisy_batch).unsqueeze(1))
                noises = Variable(torch.FloatTensor(noises).unsqueeze(1))

            # give input to the model
            output = model(inp) #returns residual
            loss = criterion(output, noises)/ (inp.size()[0]*2)
            loss.backward()
            optimizer.step()

            loss_list.append(np.float(loss.data[0]))
            loss_sum += np.float(loss.data[0])

        #get ave train loss for the epoch excluding the linear net losses
        loss_list_ave.append(loss_sum/counter)
        print('DNCNN model train loss ', loss_sum/counter)

        # compute loss test after each epoch
        model.eval()
        loss_test = 0
        for j in range(all_images['test'].shape[0]):
            output_test_ind = model(inp_test[j:j+1])
            loss_test_ind = criterion(output_test_ind, test_noises[j:j+1])/2
            loss_test += np.float(loss_test_ind.data[0])
        loss_list_test.append(loss_test/all_images['test'].shape[0])
        print( 'DNCNN model test loss ', loss_test/all_images['test'].shape[0])
        plot_loss(loss_list, loss_list_test, loss_list_ave, dir_name,h)
        output_test = model(inp_test[0:3])
        plot_denoised_image(all_params, all_images['test'][0:3], test_images_noisy[0:3], output_test )

        torch.save(model.state_dict(), dir_name  + '/model.pt')
        torch.save(model.running_sd , dir_name  + '/running_sd.pt')
        np.save(dir_name + '/loss_list.npy', loss_list)
        np.save(dir_name + '/loss_list_test.npy', loss_list_test)
        np.save(dir_name + '/loss_list_ave.npy', loss_list_ave)


    print("--- %s seconds ---" % (time.time() - start_time_total))

    # get a list of average loss for the test images for a range of noise levels

    output_test = model(inp_test[0:3])
    plot_denoised_image(all_params, all_images['test'][0:3], test_images_noisy[0:3], output_test)


    # calculate average PSNR for Test68
    psnr68_list = []
    for noise in range(0,110, 10):
        print('calculating psnr for noise  : ', noise)
        noisy68 , noise68= add_noise(all_images['test'], noise, 'S')
        if torch.cuda.is_available():
            inp68 = Variable(torch.FloatTensor(noisy68).unsqueeze(1).cuda(), volatile = True,requires_grad=False)
        else:
            inp68 = Variable(torch.FloatTensor(noisy68).unsqueeze(1), volatile = True, requires_grad=False)

        psnr68 = 0
        for i in range(inp68.size()[0]):
            output68_ind = model(inp68[i:i+1]) #gives the residual
            denoised68_ind =inp68[i:i+1] - output68_ind #gives the denosied image
            denoised68_ind_np = np.clip((denoised68_ind.data.squeeze(0).squeeze(0).cpu().numpy()) , 0,1)
            psnr68 +=  compare_psnr(denoised68_ind_np.astype('float32'), all_images['test'][i].astype('float32') ,data_range = 1)
        psnr68_list.append(psnr68/all_images['test'].shape[0])
    np.save(dir_name + '/psnr68_list.npy', psnr68_list)
    print('psnr for Test68 set over the range of 0 to 100: ', psnr68_list)

    for t in range(12):
        im = all_images['set12'][t:t+1][0]
        h,w = im.shape
        im = im.reshape(1,h,w)
        noisy12 , noise12= add_noise(im, t*10, 'S')

        if torch.cuda.is_available():
            inp12 = Variable(torch.FloatTensor(noisy12).unsqueeze(1).cuda(), volatile = True,requires_grad=False)
        else:
            inp12 = Variable(torch.FloatTensor(noisy12).unsqueeze(1), volatile = True,requires_grad=False)

        output12= model(inp12) #gives the residual

        plot_set12(all_params, all_images['set12'][t], noisy12[0], output12,t )

    return model, loss_list, loss_list_test,loss_list_ave
################################################# main #################################################

#$\frac{x + 2*padding- kernel}{stride} + 1$

def main():
    train_folder_path = '/home/zk388/berkley400_dataset/Train400/'
    test_folder_path = '/home/zk388/berkley400_dataset/Test/Set68/'
    set12_path = '/home/zk388/berkley400_dataset/Test/Set12/'

    train_folder_path = '/Users/zahra/Google Drive/Denoising_Project/my_local_learning_for_ip/2D_Signals/berkley400_dataset/Train400/'
    test_folder_path = '/Users/zahra/Google Drive/Denoising_Project/my_local_learning_for_ip/2D_Signals/berkley400_dataset/Test/Set68/'
    set12_path = '/Users/zahra/Google Drive/Denoising_Project/my_local_learning_for_ip/2D_Signals/berkley400_dataset/Test/Set12/'


    all_images= load_Berkley_dataset( train_folder_path, test_folder_path, set12_path)
    # all_images['train'] = all_images['train'][0:2]
    # all_images['test'] = all_images['test'][0:3]

    print('data is loaded')


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
    'patch_size':(40,40),
    "patching_strides" : (9,9),
    'scales' : [1,.9,.8,.7],
    }

    current_dir = os.getcwd()


    for noise_level_range in [[0,5]]:
        all_params['dir_name'] = 'noise_range_' + str(noise_level_range[0])+'to'+ str(noise_level_range[1])
        if not os.path.exists(all_params['dir_name']):
            os.makedirs(all_params['dir_name'])
        all_params['noise_level_range'] = noise_level_range
        train_entire_net(all_images, all_params)

if __name__ == "__main__" :
    main()

