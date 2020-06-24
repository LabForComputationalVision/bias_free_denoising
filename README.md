# Code and Pretrained Networks 
## "Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks"

This repository contains code and models used in the paper [Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks](https://openreview.net/pdf?id=HJlSmC4FPS) by [Sreyas Mohan*](https://sreyas-mohan.github.io), [Zahra Kadkhodaie*](https://www.linkedin.com/in/zahra-kadkhodaie-1b415680), [Eero P Simoncelli](https://www.cns.nyu.edu/~eero/) and [Carlos Fernandez-Granda](https://cims.nyu.edu/~cfgranda/), presented/published in ICLR 2020. Please visit the [project webpage here](https://labforcomputationalvision.github.io/bias_free_denoising/). [* represents equal contribution]

## Code and Pre-trained Models

Please refer to [`requirements.txt`](requirements.txt) for required packages.

### pre-trained models
The directory [`pretrained`](pretrained) contains the pretained models corresponding to DnCNN, UNet, Recurrent CNN and Simplified DenseNet (See section 5 of the [paper](https://arxiv.org/pdf/1906.05478.pdf) for more details).

### Example code for using Pre-Trained models

* In [`generalization_demo.ipynb`](generalization_demo.ipynb), we show that bias free networks generalize to noise levels outside the training range (Section 5 of the [paper](https://arxiv.org/pdf/1906.05478.pdf)).
* In [`analysis_demo.ipynb`](analysis_demo.ipynb), we examine the bias free network, visualizing the adaptive filters, and using SVD to analyze   the subspace the network is projecting onto (Section 6 of the [paper](https://arxiv.org/pdf/1906.05478.pdf)). 
The directory [`precomputed`](precomputed) contains precomputed quantities to generate various plots in the demo notebook. If required files are not present in [`precomputed`](precomputed) the notebooks will compute it and store it in the directory. 

### Train

[`train.py`](train.py) provides the code for training a model from scratch. An example usage of the script with some options is given below:

```shell
python train.py \
	--model dncnn \
	--min_noise 0 \
	--max_noise 10 \
	--data-path ./data/
```

Available models are `dncnn`, `rcnn`, `sdensenet` and `unet`. Please refer to the definition of each of these models in [`models`](models) for more options in the architecture. Adding `--bias` option to `train.py` trains the model with bias.  Please refer to the `argparse` module in [`train.py`](allcode/train.py) and [`train_utils.py`](utils/train_utils.py) for additional training options. <br>
`--data-path` expects to find `train.h5` and `valid.h5` in the folder to start training. Please refer to [pre-processing for training](#pre-processing-for-training) section for more details.

### BFBatchNorm2d

Traditional BatchNorm layers `nn.BatchNorm2d()` introduces additive constants during the mean subtraction step and addition of learned constant (`affine=True` option in PyTorch) step. We introduce a bias free version of of BatchNorm which we call BFBatchNorm. PyTorch code is provided as layer [`BFBatchNorm2d()`](models/BFBatchNorm2d.py). `BFBatchNorm2d()` can be used in place of `nn.BatchNorm2d()` in any network to make it bias free with minimal change in code. Please refer to [`dncnn.py`](models/dncnn.py) for an example usage.

### Pre-processing for training

Following [DnCNN](https://arxiv.org/abs/1608.03981) we extract patches from BSD400 data to train. 
 [`preprocess_BSD400.py`](`allcode/scripts/preprocess_BSD400.py`) provides the code to generate patches, perform data augmentation and save. The preprocessing script and data is taken from [SaoYan/DnCNN](https://github.com/SaoYan/DnCNN-PyTorch). An example usage is given below:
```shell
python preprocess_BSD400.py \
		--data_path data/ \
		--patch_size 50 \
		--stride 10 \
		--aug_times 2
```
