import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from torchvision.transforms import functional as tvf
import pandas as pd


class Compose(object):
    """
    Preprocessing class to process data in the order of 'transform'
    Both input image and annotation image are processed.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):

        width = img.size[0]  # img.size=[width][height]
        height = img.size[1]  # img.size=[width][height]

        # random scales for image expansion
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[width][height]
        scaled_h = int(height * scale)  # img.size=[width][height]

        # Resize image
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # Resize annotation
        anno_class_img = anno_class_img.resize(
            (scaled_w, scaled_h), Image.NEAREST)

        # Setting offsets based on image size
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                (left, top, left+width, top+height))

        else:
            # padding if width or height is less than input_size
            #print(anno_class_img)
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_class_img = Image.new(
                anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original,
                                 (pad_width_left, pad_height_top))
            #anno_class_img.putpalette(p_palette)

        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):

        # detemine angle of rotation
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # rotation
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror(object):
    """ Mirroring class at the probability of 50% """

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img

class Resize(object):

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        w, h = img.size
        if w != self.input_size or h != self.input_size:
            img = img.resize((self.input_size, self.input_size), Image.BICUBIC)
        w, h = anno_class_img.size
        if w != self.input_size or h != self.input_size:
            anno_class_img = anno_class_img.resize((self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img

class Resize_2(object):

    def __init__(self, input_size=1050):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        # img.size = [width][height]
        img_width = img.size[0]
        img_height = img.size[1]
        if img_height < img_width:
            img = img.resize((int((self.input_size/img_height) * img_width), self.input_size))
            anno_class_img = anno_class_img.resize(((int((self.input_size/img_height) * img_width), self.input_size)), Image.NEAREST)
        else:
            img = img.resize((self.input_size, int((self.input_size/img_width) * img_height)))
            anno_class_img = anno_class_img.resize(((self.input_size, int((self.input_size/img_width) * img_height))), Image.NEAREST)
        return img, anno_class_img

class RandomCrop(object):
    """ Crop image into input_size """

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):

        # width = img.size[0]  # img.size=[width][height]
        # height = img.size[1]  # img.size=[width][height]
        trans = transforms.RandomCrop((self.input_size, self.input_size))

        # Random cropping position
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.input_size, self.input_size))

        img_crop = tvf.crop(img, i, j, h, w) # crop input image
        anno_class_img = tvf.crop(anno_class_img, i, j, h, w) # crop annotation

        return img_crop, anno_class_img

class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):

        # PIL image into Tensor. Max value in Tensor is 1.
        img = transforms.functional.to_tensor(img)

        # Standardization
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # Annotation image in numpy
        anno_class_img = np.array(anno_class_img)  # [height][width]

        # 'ambigious' class is indexed with 0 instead of 255
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # Annotation image in Tensor
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img

class Normalize_Tensor_2(object):
    def __init__(self, color_mean, color_std, color_dict = './DeepGlobe-data/class_dict.csv'):
        self.color_mean = color_mean
        self.color_std = color_std
        self.color_dict = pd.read_csv(color_dict)

    def __call__(self, img, anno_class_img):

        # PIL image into Tensor. Max value in Tensor is 1.
        img = transforms.functional.to_tensor(img)

        # Standardization
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # Annotation image in numpy
        anno_class_img = np.array(anno_class_img)  # [height][width]

        # Transform annotation data
        #print(anno_class_img)
        #color_dict = pd.read_csv('./DeepGlobe-data/class_dict.csv')
        category_mask = np.zeros(anno_class_img.shape[:2], dtype=np.int8)
        for i, row in self.color_dict.iterrows():
            category_mask += (np.all(anno_class_img.reshape((-1, 3)) == (row['r'], row['g'], row['b']), axis=1).reshape(anno_class_img.shape[:2]) * i)
        #print(type(category_mask))
        #category_mask = Image.fromarray(category_mask)
        #print(category_mask)

        # Annotation image in Tensor
        anno_class_img = torch.from_numpy(category_mask)

        return img, anno_class_img
