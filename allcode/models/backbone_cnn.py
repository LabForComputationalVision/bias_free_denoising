import torch
import torch.nn as nn
import torch.nn.functional as F


class denoising_single_stage(nn.Module):

    def __init__(self,  kernel_size = 3, input_dimension = 1, output_dimension = 1, 
                        n_hidden = 5, hidden_dim = 64, bias = False, last_layer_bias = False):
        super(denoising_single_stage, self).__init__()

        if(kernel_size > 1):
            padding = int((kernel_size -1)/2);
        else:
            padding = 0;

        self.n_hidden = n_hidden;
        self.hidden_layer_list = [None] * (self.n_hidden);
        
        self.first_layer = nn.Conv2d(input_dimension, hidden_dim, 
                                     kernel_size, stride = 1, padding = padding,
                                     bias = bias );

        for i in range( self.n_hidden):
            self.hidden_layer_list[i] = nn.Conv2d( hidden_dim, hidden_dim, 
                                                  kernel_size, stride = 1, padding = padding,
                                                  bias = bias);
        self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list);

        self.last_layer = nn.Conv2d(hidden_dim, output_dimension, kernel_size, 
                                    stride = 1, padding = padding, bias = last_layer_bias);

    def forward(self, x):
       
        out = self.first_layer(x);
        out = F.relu(out);
        for i in range(self.n_hidden):
            out = self.hidden_layer_list[i](out);
            out = F.relu(out);

        out = self.last_layer(out);

        return(out)

