import torch
import torch.nn as nn
from models import backbone_cnn
from models import register_model


@register_model("rcnn")
class RecurrentCNN(nn.Module):
    
    def __init__(self, max_stage = 4, kernel_size = 3, n_hidden = 5,  hidden_dim = 64, 
                       bias = False,  random_n_stages = False):
        super(RecurrentCNN, self).__init__()
                
       
        self.max_stage = max_stage;
        self.random_n_stages = random_n_stages;

        input_dimension = 2;     

        self.cnn = backbone_cnn.denoising_single_stage(kernel_size = kernel_size, 
                                    input_dimension = input_dimension,
                                    n_hidden = n_hidden, 
                                    hidden_dim = hidden_dim, 
                                    bias = bias);
           

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--max_stage", type=int, default=4, help="number of stages")
        parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension")
        parser.add_argument("--num-layers", default=5, type=int, help="number of layers")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
    
    @classmethod
    def build_model(cls, args):
        return cls(max_stage = args.max_stage, hidden_dim = args.hidden_size, n_hidden = args.num_layers, bias = args.bias)

    def forward(self, x):
                                
        if self.random_n_stages or self.training:
            n_stages = np.random.randint(self.max_stage)+1;
        else:
            n_stages = self.max_stage

        prev_out = x
        for stage in range(n_stages):
            temp_input = torch.cat([ x, prev_out], dim=1);  
            prev_out = self.cnn(temp_input)

        return prev_out
