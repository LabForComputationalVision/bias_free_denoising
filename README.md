# BF_CNN
Deep Convolutional Neural Networks have produced state-of-the-art results in the problem of removing noise from images. 
These networks do not generalize well to noise levels beyond the range on which they are trained.  
But removing the additive bias terms from the networks allows 
robust generalization, and also allows us to better analyze and understand their behaviors.  In particular, we find they perform projections onto a local adaptively-estimated subspace, whose dimensionality varies with noise level.
<p>
The code in this repository implements everything shown in the paper listed below.  In particular, we provide:<br>
<UL>
<LI> Four denoising architectures, and their bias-free counterparts: 
DnCNN [Zhang et. al., IEEE Trans IP 2017];
Recurrent CNN [Zhang et al., arXiv:1805.07709 2018a];  
Unet [Ronneberger et. al., Int'l Conf Medical Image Computing, 2017]
A simplified variant of DenseNet [Huang et. al., CVPR 2017].<br>
<LI> An image dataset and code for training the networks <br>
<LI> A python notebook comparing performance of a network to its bias-free counterpart <br>
<LI> A python notebook analyzing the adaptive linear behavior of the network
</UL>
<p>
<b>Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks</b><br>
Sreyas Mohan*, Zahra Kadkhodaie*, Eero P. Simoncelli, Carlos Fernandez-Granda<br>
Presented at: <A HREF="iclr.cc">Int'l. Conf. on Learning Representations (ICLR), Apr 2020.</b><br>
Paper and reviews: https://openreview.net/forum?id=HJlSmC4FPS  <br>
Local copy: https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=MohanKadkhodaie19b <br>
Conference video and slides: https://iclr.cc/virtual/poster_HJlSmC4FPS.html 
<p>
