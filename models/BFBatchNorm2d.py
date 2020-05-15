import torch.nn as nn
import torch

class BFBatchNorm2d(nn.BatchNorm2d):
	def __init__(self, num_features, eps=1e-5, momentum=0.1, use_bias = False, affine=True):
		super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)

		self.use_bias = use_bias;

	def forward(self, x):
		self._check_input_dim(x)
		y = x.transpose(0,1)
		return_shape = y.shape
		y = y.contiguous().view(x.size(1), -1)
		if self.use_bias:
			mu = y.mean(dim=1)
		sigma2 = y.var(dim=1)

		if self.training is not True:
			if self.use_bias:        
				y = y - self.running_mean.view(-1, 1)
			y = y / ( self.running_var.view(-1, 1)**0.5 + self.eps)
		else:
			if self.track_running_stats is True:
				with torch.no_grad():
					if self.use_bias:
						self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * mu
					self.running_var = (1-self.momentum)*self.running_var + self.momentum * sigma2
			if self.use_bias:
				y = y - mu.view(-1,1)
			y = y / (sigma2.view(-1,1)**.5 + self.eps)

		if self.affine:
			y = self.weight.view(-1, 1) * y;
			if self.use_bias:
				y += self.bias.view(-1, 1)

		return y.view(return_shape).transpose(0,1)


def unit_test():

	def print_bn_details(bn):
		print(bn.running_mean)
		print(bn.running_var)

	bn_bf = BFBatchNorm2d(5, use_bias = False);
	bn_bias = BFBatchNorm2d(5, use_bias = True);
	
	print('train mode');
	bn_bf.train()
	bn_bias.train()

	for _ in range(25):
		temp_inp = torch.randn(100, 5, 128, 128)*10 + 10;
		bias_out = bn_bias(temp_inp);
		print('bias: variance %f, mean %f'%(torch.var(bias_out), torch.mean(bias_out)));
		print_bn_details(bn_bias)

		bf_out = bn_bf(temp_inp);
		print('bf: variance %f, mean %f'%(torch.var(bf_out), torch.mean(bf_out)))
		print_bn_details(bn_bf)

	print('eval mode')
	bn_bf.eval()
	bn_bias.eval()

	for _ in range(10):
		temp_inp = torch.randn(100, 5, 128, 128)*10 + 10;
		bias_out = bn_bias(temp_inp);
		print('bias: variance %f, mean %f'%(torch.var(bias_out), torch.mean(bias_out)));
		print('eval')
		print_bn_details(bn_bias)

		bf_out = bn_bf(temp_inp);
		print('bf: variance %f, mean %f'%(torch.var(bf_out), torch.mean(bf_out)))
		print('eval')
		print_bn_details(bn_bf)



