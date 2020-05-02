# Code and Pretrained Networks from <br>"Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks"

This repository contains information, code and models from the paper [Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks](https://arxiv.org/abs/1906.05478) by [Sreyas Mohan*](https://sreyas-mohan.github.io), [Zahra Kadkhodaie*](https://www.linkedin.com/in/zahra-kadkhodaie-1b415680), [Eero P Simoncelli](https://www.cns.nyu.edu/~eero/) and [Carlos Fernandez-Granda](https://cims.nyu.edu/~cfgranda/). Please visit the [project webpage here](https://sreyas-mohan.github.io/DeepFreq/). [* represents equal contribution]

## Code and Pre-trained Models

Please refer to [`requirements.txt`](requirements.txt) for required packages. 

### pre-trained models
The directory [`pretrained_models`](pretrained_models) contains the pretained models corresponding to DnCNN, UNet, Recurrent CNN and Simplified DenseNet (See section 5 of the [paper](https://arxiv.org/pdf/1906.05478.pdf) for more details).

### Example code for using Pre-Trained models

* In ['decomposition.ipynb'](decomposition.ipynb) we decompose the output of a network with bias import the linear and equivalent bias part and show that equivalent bias is small in the training range (Section 3 of the [paper](https://arxiv.org/pdf/1906.05478.pdf)).
* In [`generalization.ipynb`](generalization.ipynb), we show that bias free networks generalize to noise levels outside the training range (Section 5 of the [paper](https://arxiv.org/pdf/1906.05478.pdf)).
* In [`analysis.ipynb`](analysis.ipynb), we provide understanding of how bias free network denoising by visualizing the equivalent filters implemented by the network and analyzing the properties of the subspace the network is projecting to (Section 6 of the [paper](https://arxiv.org/pdf/1906.05478.pdf)).

### Train

[`train.py`](train.py) provides the code for training a model from scratch. An example usage of the script with some options is given below:

```shell
python train.py \
	--model dncnn \
	--min_noise 0 \
	--max_noise 10 \
  --data-path ./data/
```

Adding `--bias` option to `train.py` trains the model with bias. Available models are `dncnn`, `rcnn`, `sdensenet` and `unet`. Please refer to the definition of each of these models in [`models`](models) for more options in the architecture. Please refer to the `argparse` module in [`train.py`](train.py) and for additional training options. 

### BFBatchNorm2d

Traditional BatchNorm layers `nn.BatchNorm2d()` introduces additive consants during the mean subtraction step and addition of learned constant (`affine=True` option in PyTorch) step. We introduce a bias free version of of BatchNorm which we call BFBatchNorm. PyTorch code is provided as layer [`BFBatchNorm2d()`](models/BFBatchNorm2d.py). `BFBatchNorm2d()` can be used in place of `nn.BatchNorm2d()` in any network to make it bias free with minimal change in code. 

### Pre-processing for training

Following [DnCNN](https://arxiv.org/abs/1608.03981) we extract patches from BSD400 data to train. 
 [`preprocess_BSD400.py`](`scripts/preprocess_BSD400.py`) provides the code to generate patches, perform data augmentation and save. The preprocessing script and data is taken from [SaoYan/DnCNN](https://github.com/SaoYan/DnCNN-PyTorch). An example usage is given below:
```shell
python preprocess_BSD400.py \
    	--data_path data/train \
    	--patch_size50 \
	    --stride 10 \
   	  --aug_times 2
```
