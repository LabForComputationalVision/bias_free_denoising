# Denoising with Bias-Free CNNs

Deep Convolutional Neural Networks have produced state-of-the-art results in the problem of removing noise from images.
These networks do not generalize well to noise levels beyond the range on which they are trained. But removing the additive bias terms from the networks allows robust generalization, even when they are trained only on barely-visible levels of noise.  In addition, the removal of bias simplifies analysis of network behavior, which indicates that these denoisers perform projections onto local adaptively-estimated subspaces, whose dimensionality varies inversely with noise level.  For further information, visit the [project webpage here](https://labforcomputationalvision.github.io/bias_free_denoising/), or the published paper:

<b>Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks</b><br>
Sreyas Mohan*, Zahra Kadkhodaie*, Eero P. Simoncelli, Carlos Fernandez-Granda<br>
<A HREF="https://iclr.cc/Conferences/2020">Int'l. Conf. on Learning Representations (ICLR), Apr 2020.</A><br>

Paper (and reviews): https://openreview.net/forum?id=HJlSmC4FPS  <br>
Local copy: https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=MohanKadkhodaie19b <br>
Conference presentation (video and slides): https://iclr.cc/virtual/poster_HJlSmC4FPS.html 

## Pre-trained models

The directory [`pretrained`](pretrained) contains the pretrained models described in section 5 of the paper:
1. DnCNN [Zhang et. al., IEEE Trans IP 2017];
2. Recurrent CNN [Zhang et al., arXiv:1805.07709 2018a];
3. Unet [Ronneberger et. al., Int'l Conf Medical Image Computing, 2017];
4. A simplified variant of DenseNet [Huang et. al., CVPR 2017].<br>

For each architecture, we provide both the original model, and its bias-free counterpart. 

## Demos

We provide two Python Notebooks with example code for using pre-trained models:

* In [`generalization_demo.ipynb`](generalization_demo.ipynb), we show that bias free networks generalize to noise levels outside the training range (Section 5 of the [paper](https://arxiv.org/pdf/1906.05478.pdf)).

* In [`analysis_demo.ipynb`](analysis_demo.ipynb), we examine the bias free network, visualizing the adaptive filters, and using SVD to analyze   the subspace the network is projecting onto (Section 6 of the [paper](https://arxiv.org/pdf/1906.05478.pdf)). 

The directory [`precomputed`](precomputed) contains precomputed quantities to generate various plots in the demo notebooks. If required files are not present in [`precomputed`](precomputed) the notebooks will compute them and store them in the directory. 

Please refer to [`requirements.txt`](requirements.txt) for required Python packages.

## Training

[`train.py`](train.py) provides the code for training a model (original model, or its bias-free version) from scratch, on a provided image dataset (the Berkeley Segmentation Dataset 400).
Example usage of the script:

```shell
python train.py \
	--model dncnn \
	--min_noise 0 \
	--max_noise 10 \
	--data-path ./data/
```

Available models are `dncnn`, `rcnn`, `sdensenet` and `unet`. Please refer to the script defining each of these models in [`models`](models) for more options in the architecture. 

Adding `--bias` option to `train.py` trains the original model; otherwise, a bias-free version is trained.  Please refer to the `argparse` module in [`train.py`](train.py) and [`train_utils.py`](utils/train_utils.py) for additional training options. 

`--data-path` expects to find `train.h5` and `valid.h5` in the folder to start training. Please refer to [pre-processing for training](#pre-processing-for-training) section for more details.

### Pre-processing for training

Following [DnCNN](https://arxiv.org/abs/1608.03981) we extract patches from BSD400 data to train. 
 [`preprocess_bsd400.py`](data/preprocess_bsd400.py) provides the code to generate patches, perform data augmentation and save. The preprocessing script and data is taken from [SaoYan/DnCNN](https://github.com/SaoYan/DnCNN-PyTorch). Example usage:

```shell
python preprocess_BSD400.py \
		--data_path data/ \
		--patch_size 50 \
		--stride 10 \
		--aug_times 2
```

### BFBatchNorm2d

The traditional BatchNorm layer `nn.BatchNorm2d()` introduces additive constants during the mean subtraction step and addition of learned constant step (`affine=True` option in PyTorch). We provide PyTorch code for a bias-free BatchNorm layer [`BFBatchNorm2d()`](models/BFBatchNorm2d.py), which eliminates the subtraction/addition steps (i.e., it only divides by standard deviation).  This can be used in place of `nn.BatchNorm2d()` to produce a bias-free version of a network. Please refer to [`dncnn.py`](models/dncnn.py) for example usage.
