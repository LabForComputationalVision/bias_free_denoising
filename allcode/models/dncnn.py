import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
from models import BFBatchNorm2d

@register_model("dncnn")
class DnCNN(nn.Module):
    """DnCNN as defined in https://arxiv.org/abs/1608.03981 """
    def __init__(self, depth=20, n_channels=64, image_channels=1, bias=False, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1

        self.bias = bias;
        if not bias:
            norm_layer = BFBatchNorm2d.BFBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.depth = depth;


        self.first_layer = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)

        self.hidden_layer_list = [None] * (self.depth - 2);
        
        self.bn_layer_list = [None] * (self.depth -2 );
        
        for i in range(self.depth-2):
            self.hidden_layer_list[i] = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias);
            self.bn_layer_list[i] = norm_layer(n_channels)
        
        self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list);
        self.bn_layer_list = nn.ModuleList(self.bn_layer_list);
        self.last_layer = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)
     
        self._initialize_weights()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--in-channels", type=int, default=1, help="number of channels")
        parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension")
        parser.add_argument("--num-layers", default=20, type=int, help="number of layers")
        parser.add_argument("--bias", action='store_true', help="use residual bias")

    @classmethod
    def build_model(cls, args):
        return cls(image_channels = args.in_channels, n_channels = args.hidden_size, depth = args.num_layers, bias=args.bias)

    def forward(self, x):
        y = x
        out = self.first_layer(x);
        out = F.relu(out);

        for i in range(self.depth-2):
            out = self.hidden_layer_list[i](out);
            out = self.bn_layer_list[i](out);
            out = F.relu(out)

        out = self.last_layer(out);
        
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

