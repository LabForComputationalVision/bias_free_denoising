import torch

def get_noise(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    noise = torch.randn_like(data);
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
    else:
        noise.data = noise.data * noise_std;
    return noise