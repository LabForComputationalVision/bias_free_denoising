import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import os
import time
import torch
from torch.autograd import Variable
from skimage import io


def make_stepEdge(sides_diff,  im_shape):
    '''@im_shape: tuple of two even numbers
    @sides_diff: float between 0 and 1'''
    sides_diff = np.clip(sides_diff, 0,1)
    im = np.zeros(im_shape)
    if np.random.randint(2) == 0:
        im[:,int(im_shape[0]/2):: ] = .5 + (sides_diff/2)
        im[:,0:int(im_shape[0]/2) ] = .5 - (sides_diff/2)
    else:
        im[:,int(im_shape[0]/2):: ] = .5 - (sides_diff/2)
        im[:,0:int(im_shape[0]/2) ] = .5 + (sides_diff/2)
    return im




def calc_jacobian( im,model, all_params):


    ############## Prepare Variables
    if torch.cuda.is_available():
        inp = Variable(torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).cuda(),requires_grad=True)
    else:
        inp = Variable(torch.FloatTensor(im).unsqueeze(0).unsqueeze(0),requires_grad=True)

    ############## prepare the static model
    # model = Net(all_params)
    for param in model.parameters():
        param.requires_grad = False


    model.eval()


    ############## find Jacobian
    out = model(inp)
    jacob = []
    for i in range(inp.size()[2]):
        for j in range(inp.size()[3]):
            part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True) # this gives me a 20*20
            jacob.append( part_der[0][0,0].data.view(-1)) # flatten it to 400


    return torch.stack(jacob)

def plot_correlations(jacob, clean_im, noise, all_params):
    im_shape = clean_im.shape
    I = np.eye(int(im_shape[0]*im_shape[1]))
    U,S, V = np.linalg.svd(I - jacob.cpu().numpy())
    n = S.shape[0]

    im_shape = clean_im.shape
    noise_corr_u = []
    im_clean_corr_u = []
    noise_corr_v = []
    im_clean_corr_v = []
    for i in range(0,n):
        v = V[i,:]
        u = U[:,i]
        noise_corr_u.append(np.dot(u,noise.flatten())/np.linalg.norm(noise) )
        im_clean_corr_u.append(np.dot(u,(clean_im).flatten())/np.linalg.norm(clean_im) )

        noise_corr_v.append(np.dot(v,noise.flatten())/np.linalg.norm(noise) )
        im_clean_corr_v.append(np.dot(v,(clean_im).flatten())/np.linalg.norm(clean_im) )


    plt.figure(figsize = (12,12))
    plt.plot(S,  '.', label = 'singular values')
    plt.axhline(1, color = 'k' )
    plt.axhline(0, color = 'r', alpha=.6 )

    plt.plot(noise_corr_u,  '.' , label = 'corr bw u and noise')
    plt.plot(im_clean_corr_u,  '.', label = 'corr bw u and sig')
    # plt.plot(noise_corr_v,  '.' , label = 'corr bw v and noise')
    # plt.plot(im_clean_corr_v,  '.', label = 'corr bw v and sig')
    plt.legend();
    plt.savefig(all_params['dir_name'] +'/'+all_params['folder_name']  + '/sig_noise_sing_corr_U.png')


    plt.figure(figsize = (12,12))
    plt.plot(S,  '.', label = 'singular values')
    plt.axhline(1, color = 'k' )
    plt.axhline(0, color = 'r', alpha=.6 )

    # plt.plot(noise_corr_u,  '.' , label = 'corr bw u and noise')
    # plt.plot(im_clean_corr_u,  '.', label = 'corr bw u and sig')
    plt.plot(noise_corr_v,  '.' , label = 'corr bw v and noise')
    plt.plot(im_clean_corr_v,  '.', label = 'corr bw v and sig')
    plt.legend();
    plt.savefig(all_params['dir_name'] +'/'+all_params['folder_name']  + '/sig_noise_sing_corr_V.png')


def plot_uv_dots(jacob, all_params):


    U,S, V = np.linalg.svd( jacob.cpu().numpy())

    n = S.shape[0]

    uvdots = []
    for i in range(0,n):
        v = V[i,:]
        u = U[:,i]
        uvdots.append(np.dot(v,u))

    plt.figure(figsize = (12,12))
    plt.plot(S,  '.', label = 'singular values')
    plt.plot(uvdots, '.', label = 'cos angel between u and v')
    plt.axhline(1, color = 'k' )
    plt.axhline(0, color = 'r', alpha=.6 )
    plt.axhline(-1, color = 'k' , alpha=.6 )
    plt.legend();
    plt.savefig(all_params['dir_name'] +'/'+all_params['folder_name']  + '/uvdots.png')

def calc_singular_vectors( im ,im_denoised, jacob ,all_params,k, res, plot = True ):
    '''@k: a sequence of singular vectors indices '''
    im_shape = im.shape
    I = np.eye(int(im_shape[0]*im_shape[1]))
    U,S, V = np.linalg.svd(I - jacob.cpu().numpy())


    np.save(all_params['dir_name'] +'/' +all_params['folder_name']  + '/singular_values.npy', S)

    if plot is True:
        n = S.shape[0]

        plt.figure(figsize = (12,12))
        plt.plot(S,  'o')
        plt.title('singular values of I - J')
        plt.savefig(all_params['dir_name'] +'/'+all_params['folder_name']  + '/singular_values.png')



        # proj_comb = np.zeros(im_shape)

        for i in k:
            v = V[i,:]
            u = U[:,i]
            f , axs = plt.subplots(2,3 , figsize=(18,12))
            f.suptitle('I - J ',  fontsize = 20)
            ## plot the left s vect
            # limit =  max(np.abs(np.min( u)), np.abs(np.max(u)))
            fig = axs[0,0].imshow(u.reshape(im_shape), 'gray')
            plt.colorbar(fig, ax=axs[0,0], fraction=.05)
            axs[0,0].set_title('u assoc. with \n '+str(i)+'th s: '+ str(np.round(S[i] ,3) ))

            ## plot the right s vect
            # limit =  max(np.abs(np.min( v)), np.abs(np.max(v)))
            fig = axs[0,1].imshow(v.reshape(im_shape), 'gray')
            plt.colorbar(fig, ax=axs[0,1], fraction=.05)
            axs[0,1].set_title('v assoc. with \n '+str(i)+'th s: '+ str(np.round(S[i] ,3) ))

            ## plot the projection of image on the space of v and u
            proj = np.dot(np.dot(u.reshape(n,1), v.reshape(1,n)), im.flatten()).reshape(im_shape)
            # limit =  max(np.abs(np.min( proj)), np.abs(np.max(proj)))
            fig = axs[0,2].imshow(proj, 'gray')
            plt.colorbar(fig, ax = axs[0,2], fraction=.05)
            axs[0,2].set_title(' u_i . v_i . y')



            ## plot the noisy image
            fig = axs[1,0].imshow(im,  'gray', vmin = 0, vmax = 1)
            plt.colorbar(fig, ax = axs[1,0], fraction=.05)
            axs[1,0].set_title('noisy image\n norm '+str(np.round(np.linalg.norm(im) ,3)))

            ## plot the denoised image
            fig = axs[1,1].imshow(im_denoised,  'gray', vmin = 0, vmax = 1)
            plt.colorbar(fig, ax = axs[1,1], fraction=.05)
            axs[1,1].set_title('denoised image \n norm '+str(np.round(np.linalg.norm(im_denoised) ,3)))

            # ## projections scaled

            fig = axs[1,2].imshow( S[i] * proj , 'gray', vmin = 0, vmax = 1)
            plt.colorbar(fig, ax = axs[1,2], fraction=.05)
            axs[1,2].set_title('s_i . u_i . v_i . y \n norm ' +str(np.round(np.linalg.norm(S[i] * proj ) ,3)))


            plt.savefig(all_params['dir_name'] +'/'+all_params['folder_name']+'/'  + str(i) + 'th_distortion.png')
            plt.close('all')


        ## add the projections in between
        # proj_comb = proj_comb+ np.dot(np.dot(U[:,k:n-k+5], np.dot(np.diag(S[k:n-k+5]), V[k:n-k+5 , : ]) ), im.flatten()).reshape(im_shape)


    return U, S, V

