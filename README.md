# pp-pad
This repository contains the pytorch implementation of pp-pad (peripheral prediction padding) in ICIP2023 paper:

- Kensuke Mukai and Takao Yamanaka, "Improving Translation Invariance in Convolutional Neural Networks with Peripheral Prediction Padding," International Conference on Image Processing (ICIP), 2023, Kuala Lumpur, Malaysia. [arXiv https://arxiv.org/abs/2307.07725]

![outline of pp-pad](samples/outline_pp-pad.png)

# Downloads

# Examples
Change current directory:
> cd program

For training the network:
> python PSPNet.py --cfg config/cfg_sample.yaml

For evaluate the translation invarinace and classification accuracy:
> python segment_val.py --cfg config/cfg_sample.yaml

The results are saved in the 'outputs' directory. The filepath and other settings are specified in the config file 'config/*.yaml'.

# Settings in Config (program/config)
The yaml file can be modified in a text editor.  
sample config file: program/config/cfg_sample.yaml

[padding mode]
- padding_mode: select from 'pp-pad', 'zeros', 'reflect', 'replicate', 'circular', 'partial', 'cap_pretrain', or 'cap_train'. For 'cap_pretrain' and 'cap_train', use them in this order, since pretrain is required in CAP.

[path for PSPNet.py]
- dataset: dataset folder
- pretrain: initial weights for the network
- outputs: output folder

[training parameters]
- num_epoches: Number of training epochs
- val_output_interval: interval for validation [epochs]
- batch_size
- batch_multiplier: weights are updated every batch_multiplier, which means the effective batch size is batch_size x batch_multiplier
- optimizer: sgd or adam

[path for segment_val.py]
- val_images: validation file list for evaluation of translation invariance and classification accuracy (only 100 images were used for evaluation due to the computational cost)
- weights: network model for evaluation

[image sizes]
- input_size: patch size extracted from original image in training and evaluation
- expanded_size: original image was first resized into expanded_size in the long side, and then croped in the input_size specified above.
- patch_stride: stride of sliding window in evaluation creating overlapping patches

[dataset info]
- color_mean: mean values of images in the dataset
- color_std: standard deviations of images in the dataset

# Results
The values in the following results were different from the original ICIP2023 paper[1], especially in meanE, because there were bugs in the initial implementation for calculating meanE. The following is the results obatained by the current code.

[Simple mean IOU, meanE & disR including background class]
- mIoU is simple mean, but not weighted average. The background class was excluded in the calculation of IoU.
- meanE and disR were calculated including background class.

| | Methods | mIoU &uarr; | meanE_in &darr;| disR_in &darr; |
| ---- | ---- | ---- | ---- | ---- |
| Conventional | Zero | 0.3234 | 0.4786 | 0.6151 |
| | Reflect |
| | Replicate |
| | Circular |
| Previous | CAP [19] |
| | Partial [17] |
| Proposed | PP-Pad (2x3) |
| | PP-Pad (3x3) |
| | PP-Pad (5x3) |
| | PP-Pad (10x3) |

[Weighted average IOU, meanE & disR excluding background class]

- Weighted average version of mIoU (mIoU_weighted)
- meanE & disR excluding the background class (meanE_ex, disR_ex)

| | Methods | mIoU_weighted &uarr; | meanE_ex &darr; | disR_ex &darr; |
| ---- | ---- | ---- | ---- | ---- |
| Conventional | Zero | 0.4295 | 0.6012 | 0.7515 |
| | Reflect |
| | Replicate |
| | Circular |
| Previous | CAP [19] |
| | Partial [17] |
| Proposed | PP-Pad (2x3) |
| | PP-Pad (3x3) |
| | PP-Pad (5x3) |
| | PP-Pad (10x3) |

# References
1. Kensuke Mukai and Takao Yamanaka, "Improving Translation Invariance in Convolutional Neural Networks with Peripheral Prediction Padding," ICIP2023. https://arxiv.org/abs/2307.07725
2. Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia, "Pyramid Scene Parsing Network," CVPR2017. https://arxiv.org/abs/1612.01105
3. Yu-Hui Huang, Marc Proesmans, and Luc Van Gool, "Context-aware Padding for Semantic Segmentation," arXiv 2021. https://arxiv.org/abs/2109.07854
4. Guilin Liu, Kevin J. Shih, Ting-Chun Wang, Fitsum A. Reda, Karan Sapra, Zhiding Yu, Andrew Tao, and Bryan Catanzaro, "Partial Convolution based Padding," arXiv 2018. https://arxiv.org/abs/1811.11718

# Versions
The codes were confired with the following versions.
- Python 3.7.13
- Pytorch 1.13.0+cu117
- NVIDIA Driver 510.108.03
- CUDA 11.6