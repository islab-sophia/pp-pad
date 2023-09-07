import os.path as osp
from PIL import Image
import torch.utils.data as data
import pandas as pd
import os
import torch
import cv2
import numpy as np

from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Resize_2, Normalize_Tensor, RandomCrop

DEFAULT_EXPANDED_SIZE = 1050
DEFAULT_INPUT_SIZE = 475
DEFAULT_COLOR_MEAN = (0.485, 0.456, 0.406)
DEFAULT_COLOR_STD = (0.229, 0.224, 0.225)

def make_datapath_list(rootpath):
    """
    Create filepath list to image and annotation data in train and val

    Parameters
    ----------
    rootpath : str
        Path to dataset
    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        Filepath list 
    """

    # String template for filepath of image file or annotation file
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    # Filepath including train filenames and val filenames
    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # Create filepath lists for train image files and annotation files
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # Removing spaces and return codes
        img_path = (imgpath_template % file_id)  # image file path
        anno_path = (annopath_template % file_id)  # annotation file path
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # Create filepath lists for validation image files and annotation files
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # Removing spaces and return codes
        img_path = (imgpath_template % file_id)  # image file path
        anno_path = (annopath_template % file_id)  # annotation file path
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


def make_datapath_list_dgd(rootpath):

    DATA_DIR = rootpath + 'train/'

    metadata_df = pd.read_csv(osp.join(rootpath, 'metadata.csv'))
    metadata_df = metadata_df[metadata_df['split']=='train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    # Shuffle DataFrame
    #metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    # Perform 90/10 split for train / val
    valid_df = metadata_df.sample(frac=0.1, random_state=0)
    train_df = metadata_df.drop(valid_df.index)

    train_img_list = list()
    train_anno_list = list()
    val_img_list = list()
    val_anno_list = list()
    
    return train_img_list, train_anno_list, val_img_list, val_anno_list

class DataTransform():
    """
    Preprocessing Class for Input Image and Annotation for each mode (train or val)
    Output size: input_size x input_size
    Data augmentation only for train

    Attributes
    ----------
    input_size : int
        Output image size
    color_mean : (R, G, B)
        mean value for each channel
    color_std : (R, G, B)
        standard deviation for each channel
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # expansion
                RandomRotation(angle=[-10, 10]),  # rotation
                RandomMirror(),  # mirror
                Resize(input_size),  # resize into image_size
                Normalize_Tensor(color_mean, color_std)  # standardization
            ]),
            'val': Compose([
                Resize(input_size),  # resize into image_size
                Normalize_Tensor(color_mean, color_std)  # standardization
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            Preprocessing mode
        """
        return self.data_transform[phase](img, anno_class_img)

class DataTransform_2():
    """
    Preprocessing Class for Input Image and Annotation for each mode (train or val)
    Output size: input_size x input_size
    Data augmentation only for train
    The image is expanded into expanded_size before cropping into image_size.

    Attributes
    ----------
    input_size : int
        Output image size
    expanded_size : int
        Intermediate image size before cropping into input_size
    color_mean : (R, G, B)
        mean value for each channel
    color_std : (R, G, B)
        standard deviation for each channel
    """

    def __init__(self, input_size=DEFAULT_INPUT_SIZE, expanded_size=DEFAULT_EXPANDED_SIZE, color_mean=DEFAULT_COLOR_MEAN, color_std=DEFAULT_COLOR_STD):
        self.data_transform = {
            'train': Compose([
                Resize_2(expanded_size),  # resize to expanded_size (expansion)
                RandomCrop(input_size),   # crop into image_size
                Scale(scale=[0.5, 1.5]),  # expansion
                RandomRotation(angle=[-10, 10]),  # rotation
                RandomMirror(),  # mirror
                Resize(input_size),  # resize into image_size
                Normalize_Tensor(color_mean, color_std)  # standardization
            ]),
            'val': Compose([
                Resize_2(expanded_size),  # resize into image_size
                RandomCrop(input_size),
                Resize(input_size),  # resize into image_size
                Normalize_Tensor(color_mean, color_std)  # standardization
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            Preprocessing mode
        """
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    """
    VOC2012 Dataset Class

    Attributes
    ----------
    img_list : List
        List of image file paths
    anno_list : List
        List of annotation file paths
    phase : 'train' or 'test'
        training or test mode
    transform : object
        Instance of preprocessing class
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''Return the number of images'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        Get image data and annotation data in Tersor, which are preprocessed with transform()
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''To get image data and annotation data in Tensor'''

        # 1. Load image
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # img.size = [width][height]

        # 2. Load annotation file
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # anno_class_img.size = [width][height]

        # 3. Preprocessing with transform()
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


class DeepGlobeDataset(torch.utils.data.Dataset):

    """
    DeepGlobe Land Cover Classification Challenge Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    def __init__(
            self, 
            df,
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)