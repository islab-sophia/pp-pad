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
- padding_mode: select from 'zeros', 'reflect', 'replicate', 'circular', 'partial', 'cap_pretrain', 'cap_train' or 'pp-pad'

[path for PSPNet.py]
- dataset: dataset folder
- pretrain: initial weights for the network
- outputs: output folder

[training parameters]
- num_epoches: Number of training epochs
- val_output_interval: interval for validation [epochs]
- batch_size
- batch_multiplier: weights are updated every batch_multiplier, which means the effective batch size is batch_size x batch_multiplier
- optimizer

[path for segment_val.py]
- val_iamges: filepath for evaluation of translation invariance and classification accuracy (only 100 images were used for evaluation due to the computational cost)
- weights: network model for evaluation

[image sizes]
- input_size: patch size extracted from original image in training and evaluation
- expanded_size: original image was resized into expanded_size in the long side, and then croped in the input_size specified above.
- patch_stride: stride of sliding window in evaluation

[dataset info]
- color_mean: mean value of images in the dataset
- color_std: standard deviation of images in the dataset

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