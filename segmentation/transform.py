import random
from PIL import Image
import math
import numbers
from collections.abc import Iterable
import warnings
import types
import collections

import cv2
# import mmcv
import numpy as np

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance

_cv2_pad_to_str = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT
}
INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}


class Compose:
    """Composes several transforms together.
    Args:
        transforms(list of 'Transform' object): list of transforms to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):

        for trans in self.transforms:
            img, mask = trans(img, mask)

        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize:
    """Resize an image and an mask to given size
    Args:
        size: expected output size of each edge, can be int or iterable with (h, w)
        if range is given:
            resize image to size  [h * s, w * s],  s is sampled from range .
    """

    def __init__(self, size=None, range=None):

        if size is not None:
            if isinstance(size, int):
                self.size = (size, size)
            elif isinstance(size, Iterable) and len(size) == 2:
                self.size = size
            else:
                raise TypeError('size should be iterable with size 2 or int')

        elif range is not None:
            if isinstance(range, Iterable) and len(range) == 2:
                self.range = range
            else:
                raise TypeError('size should be iterable with size 2 or int')

        # assert  range is not None and size is None
        if not (range is None) ^  (size is None):
            raise ValueError('can not both be set or not set')

        self.size = size
        self.range = range

    def __call__(self, img, mask):

        if self.range:
            ratio = random.uniform(*self.range)
            resized_img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
            resized_mask = cv2.resize(mask, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

        if self.size:
            h, w = self.size
            size = (w, h)
            resized_img = cv2.resize(img, size)
            resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        # print(np.unique(resized_mask))
        return resized_img, resized_mask

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i + h, j: j+ w, ...]

def center_crop(img, output_size, fill=0):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    h, w = img.shape[0:2]
    th, tw = output_size
    pad_left = max(int((tw - w) / 2), 0)
    pad_right = max(tw - w - pad_left, pad_left)
    pad_top = max(int((th - h) / 2), 0)
    pad_bot = max(th - h - pad_top, pad_top)
    img = pad(img, (pad_left, pad_top, pad_right, pad_bot), fill=fill)
    h, w = img.shape[0:2]

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given numpy ndarray on all sides with specified padding mode and fill value.
    Args:
        img (numpy ndarray): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value io only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        Numpy image: padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(
            type(img)))
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding,
                  collections.abc.Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.abc.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.abc.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    return cv2.copyMakeBorder(img,
                              top=pad_top,
                              bottom=pad_bottom,
                              left=pad_left,
                              right=pad_right,
                              borderType=_cv2_pad_to_str[padding_mode],
                              value=fill)

class RandomCrop(object):
    """Crop the given numpy ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self,
                #  size,
                crop_size,
                #  padding=None,
                pad_if_needed=False,
                pad_value=0,
                seg_pad_value=255,
                cat_max_ratio=1.):
                #  padding_mode='constant'):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        # self.padding = padding
        self.pad_if_needed = pad_if_needed
        # self.fill = fill
        # self.padding_mode = padding_mode
        self.pad_value= pad_value
        self.seg_pad_value = seg_pad_value
        #self.ignore_value = ignore_value,
        self.cat_max_ratio = cat_max_ratio

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        # print(img.shape[:2], output_size)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, max(h - th, 0))
        j = random.randint(0, max(w - tw, 0))
        return i, j, th, tw

    def __call__(self, img, mask):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        #if self.padding is not None:
        #    img = pad(img, self.padding, self.pad_value, self.padding_mode)
        #    mask = pad(mask, self.padding, self.seg_pad_value, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.crop_size[1]:
        # if self.pad_if_needed and img.shape[1] < self.crop_size[1]:
            left_pad = int((self.crop_size[1] - img.shape[1]) / 2)
            right_pad = self.crop_size[1] - img.shape[1] - left_pad
            #img = pad(img, (self.size[1] - img.shape[1], 0), self.fill,
            #            self.padding_mode)
            img = pad(img, (left_pad, 0, right_pad, 0), self.pad_value,
                        # self.padding_mode)
                        'constant')
            #mask = pad(mask, (self.size[1] - mask.shape[1], 0), self.fill,
            #            self.padding_mode)
            mask = pad(mask, (left_pad, 0, right_pad, 0), self.seg_pad_value,
                        # self.padding_mode)
                        'constant')
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.crop_size[0]:
            top_pad = int((self.crop_size[0] - img.shape[0]) / 2)
            bot_pad = self.crop_size[0] - img.shape[0] - top_pad
            #img = pad(img, (0, self.size[0] - img.shape[0]), self.fill,
            #            self.padding_mode)
            #mask = pad(mask, (0, self.size[0] - mask.shape[0]), self.fill,
            #            self.padding_mode)
            img = pad(img, (0, top_pad, 0, bot_pad), self.pad_value,
                        'constant')
            mask = pad(mask, (0, top_pad, 0, bot_pad), self.seg_pad_value,
                        # self.padding_mode)
                        'constant')

        #print(self.pad_if_needed)
        i, j, h, w = self.get_params(img, self.crop_size)
        # print(i,j,h,w)

        # for self.
        # mask = crop(mask, i, j, h, w)

        #for
        if self.cat_max_ratio < 1:
            for _ in range(10):
                # print(_)
                mask_temp = crop(mask, i, j, h, w)
                labels, cnt = np.unique(mask_temp, return_counts=True)
                cnt = cnt[labels != self.seg_pad_value]

                thresh = np.sum(cnt) / (mask_temp.shape[0] * mask_temp.shape[1])
                # print(thresh)
                if thresh < 0.75:
                    continue

                if len(cnt) > 1 and np.max(cnt) / np.sum(
                    cnt) < self.cat_max_ratio:
                    break

                i, j, h, w = self.get_params(img, self.crop_size)



        img = crop(img, i, j, h, w)
        mask = crop(mask, i, j, h, w)


        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(
            self.crop_size)

def rotate(img, angle, resample='BILINEAR', expand=False, center=None, value=0):
    """Rotate the image by angle.
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): In degrees clockwise order.
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """
    imgtype = img.dtype
    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    #h, w, _ = img.shape
    h, w = img.shape[:2]
    point = center or (w/2, h/2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(img, M, (nW, nH), flags=INTER_MODE[resample], borderValue=value)
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w-1, 0, 1]), np.array([w-1, h-1, 1]), np.array([0, h-1, 1])):
                target = M@point
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w)/2
            M[1, 2] += (nh - h)/2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=INTER_MODE[resample], value=value)
    else:
        dst = cv2.warpAffine(img, M, (w, h), flags=INTER_MODE[resample], value=value)
    return dst.astype(imgtype)

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        resample ({CV.Image.NEAREST, CV.Image.BILINEAR, CV.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample='BILINEAR', expand=False, center=None, pad_value=0, seg_pad_value=255):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, mask):
        """
            img (np.ndarray): Image to be rotated.
        Returns:
            np.ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)

        img = rotate(img, angle, self.resample, self.expand, self.center, value=self.pad_value)
        # print(np.unique(mask))
        mask = rotate(mask, angle, 'NEAREST', self.expand, self.center, value=self.seg_pad_value)

        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomVerticalFlip:
    """Horizontally flip the given opencv image with given probability p.
    and does the same to mask

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() <= self.p:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        return img, mask

class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.
    and does the same to mask

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() <= self.p:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        return img, mask

class RandomGaussianBlur:
    """Blur an image using gaussian blurring.

    Args:
       sigma: Standard deviation of the gaussian kernel.
       Values in the range ``0.0`` (no blur) to ``3.0`` (strong blur) are
       common. Kernel size will automatically be derived from sigma
       p: probability of applying gaussian blur to image

       https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmenters/blur.html#GaussianBlur
    """

    def __init__(self, p=0.5, sigma=(0.0, 3.0)):

        if not isinstance(sigma, Iterable) and len(sigma) == 2:
            raise TypeError('sigma should be iterable with length 2')

        if not sigma[1] >= sigma[0] >= 0:
            raise ValueError(
                'sigma shoule be an iterval of nonegative real number')

        self.sigma = sigma
        self.p = p

    def __call__(self, img, mask):

        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            k_size = self._compute_gaussian_blur_ksize(sigma)
            img = cv2.GaussianBlur(img, (k_size, k_size),
                                   sigmaX=sigma, sigmaY=sigma)

        return img, mask

    @staticmethod
    def _compute_gaussian_blur_ksize(sigma):
        if sigma < 3.0:
            ksize = 3.3 * sigma  # 99% of weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # 97% of weight
        else:
            ksize = 2.6 * sigma  # 95% of weight

        ksize = int(max(ksize, 3))

        # kernel size needs to be an odd number
        if not ksize % 2:
            ksize += 1

        return ksize

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, mask):
        return self.lambd(img), mask

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch
    float tensor (c, h, w) ranged from 0 to 1, and convert mask to torch tensor
    """

    def __call__(self, img, mask):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]

        Returns:
            a pytorch tensor
        """
        #convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        mask = torch.from_numpy(mask).long()

        return img, mask

class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    and does nothing to mask tensor

    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img, mask):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        if img.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(img.size()))

        if not self.inplace:
            img = img.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)

        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        img.sub_(mean).div_(std)

        # img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img, mask


class RandomScaleCrop:
    """Randomly scaling an image (from 0.5 to 2.0]), the output image and mask
    shape will be the same as the input image and mask shape. If the
    scaled image is larger than the input image, randomly crop the scaled
    image.If the scaled image is smaller than the input image, pad the scaled
    image.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        value: value to fill the mask when resizing,
               should use ignore class index
    """

    def __init__(self, crop_size, scale=(0.5, 2.0), value=0, padding_mode='constant'):

        if not isinstance(scale, Iterable) and len(scale) == 2:
            raise TypeError('scale should be iterable with size 2 or int')

        self.fill = value
        self.scale = scale
        self.crop_size = crop_size
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, mask):

        scale = random.uniform(self.scale[0], self.scale[1])

        crop_size = int(self.crop_size / scale)

        if img.shape[1] < crop_size:
            left_pad = int((crop_size - img.shape[1]) / 2)
            right_pad = crop_size - img.shape[1] - left_pad
            img = pad(img, (left_pad, 0, right_pad, 0), 0,
                        self.padding_mode)
            mask = pad(mask, (left_pad, 0, right_pad, 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if img.shape[0] < crop_size:
            top_pad = int((crop_size - img.shape[0]) / 2)
            bot_pad = crop_size - img.shape[0] - top_pad
            img = pad(img, (0, top_pad, 0, bot_pad), 0,
                        self.padding_mode)
            mask = pad(mask, (0, top_pad, 0, bot_pad), self.fill,
                        self.padding_mode)

        i, j, h, w = self.get_params(img, (crop_size, crop_size))
        img = crop(img, i, j, h, w)
        mask = crop(mask, i, j, h, w)

        img = cv2.resize(img, (self.crop_size, self.crop_size))
        mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)


        return img, mask

class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        #if random.randint(2):
        if random.randint(0, 1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 1):
            # img = mmcv.bgr2hsv(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            #img = mmcv.hsv2bgr(img)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 1):
            # img = mmcv.bgr2hsv(img)
            # img = mmcv.bgr2hsv(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            # img = mmcv.hsv2bgr(img)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        # img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        #mode = random.randint(2)
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        #results['img'] = img
        #return results
        return img, gt_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

class CenterCrop(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.fill = fill

    def __call__(self, img, mask):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        return center_crop(img, self.size, fill=self.fill), center_crop(mask, self.size, fill=self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)



#img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/img.jpg'
#img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_16.bmp'
#
#crop_size=(480, 480)
#trans = Resize(range=[0.5, 1.5])
#trans1 = RandomRotation(degrees=90, expand=True)
#trans2 = RandomCrop(crop_size=crop_size, cat_max_ratio=0.75, pad_if_needed=True)
#trans3 = RandomVerticalFlip()
#trans4 = RandomHorizontalFlip()
#
##trans = Compose([
##    Resize(range=[0.5, 1.5]),
##    RandomRotation(degrees=90, expand=True, pad_value=0, seg_pad_value=255),
##])
## trans5 = ColorJitter(p=0, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
#trans5 = PhotoMetricDistortion()
#
#img = cv2.imread(img_path)
#
#print(img.shape)
#
#import time
## start = time.time()
## for _ in range(1000):
#    # img, mask = trans5(img, img[:, :, 0])
#    # _ = trans5(img, img[:, :, 0])
## finish = time.time()
#img, mask = trans5(img, img[:, :, 0])
#img, mask = trans(img, mask)
#img, mask = trans1(img, mask)
#img, mask = trans2(img, mask)
#img, mask = trans3(img, mask)
#img, mask = trans4(img, mask)
## print(finish - start)
## print(img.shape, mask.shape)
#
#print(img.shape, mask.shape)
#cv2.imwrite('src.jpg', img)
#cv2.imwrite('src1.jpg', mask)


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        # _log_api_usage_once(self)
        self.transforms = transforms
        self.p = p

    def forward(self, img, mask):
        if random.random() < self.p:
        # if self.p < torch.rand(1):
            return img, mask

        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string




class MultiScaleFlipAug(object):
    """Return a set of MultiScale fliped Images"""


    def __init__(self,
                #  transforms,
                #  img_scale,
                # img_ratios=None,
                 img_ratios,
                 mean,
                 std,
                 transforms=None,
                 flip=False,
                 flip_direction='horizontal'):

        img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]

        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]


        # normalize and to_tensor
        self.transforms = transforms

        for flip_direction in self.flip_direction:
            assert flip_direction in ['vertical', 'horizontal']

        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        # print(self.flip_direction, 'cccccccccccccccccccccccccc')

    def construct_flip_param(self):

        flip_aug = [False, True] if self.flip else [False]
        if len(self.flip_direction) == 2:
            flip_aug.append(True)


        flip_direction = self.flip_direction.copy()
        if self.flip:
            flip_direction.append(flip_direction[0])

        # print(flip_aug, flip_direction)
        assert len(flip_aug) == len(flip_direction)

        return list(zip(flip_aug, flip_direction))


    def norm(self, img):

        std = self.std
        mean = self.mean

        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return img.sub_(mean).div_(std)

    def mask_to_tensor(self, mask):
        mask = torch.from_numpy(mask).long()
        return mask


    def img_to_tensor(self, img):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        return img


    def __call__(self, img, gt_seg):
        """Call function to apply test time augment transforms on results.
        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        img_meta = {
            "seg_map": None,
            "imgs" : [],
            "flip" : []
        }

        flip_param = self.construct_flip_param()
        print('before:', type(gt_seg))

        for ratio in self.img_ratios:

            resized_img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

            for flip, direction in flip_param:

                if flip:

                    if direction == 'horizontal':
                        flipped_img = cv2.flip(resized_img, 1)
                        img_meta['flip'].append(direction)

                    if direction == 'vertical':
                        flipped_img = cv2.flip(resized_img, 0)
                        img_meta['flip'].append(direction)

                else:
                    img_meta['flip'].append('none')
                    flipped_img = resized_img

                img_tensor = self.img_to_tensor(flipped_img)
                norm_img = self.norm(img_tensor)

                # normalize + to_tensor
                # if self.transforms is not None:
                    # for trans in self.transforms:
                        # print(type(gt_seg), trans)
                        # flipped_img, gt_seg = trans(flipped_img, gt_seg)
                img_meta['imgs'].append(norm_img)

        img_meta['seg_map'] = self.mask_to_tensor(gt_seg)
        # print(img_meta['flip'])
        # import sys; sys.exit()

        return img_meta



# trans = MultiScaleFlipAug(
#     img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
#     #img_ratios=[0.5],
#     transforms=None,
#     flip=True,
#     flip_direction=['horizontal', 'vertical'],
# )

# # img = cv2.imread('/data/hdd1/by/Self_Contained_OpenMMLab/Lenna_(test_image).png')
# img = cv2.imread('/data/hdd1/by/Self_Contained_OpenMMLab/LENWe.jpg')

# img_meta = trans(img, img)

# #print(img_meta)
# flip = img_meta['flip']
# count = 0
# for key, value in img_meta.items():
#     #print(key, value)
#     if key == 'imgs':

#         for img in value:
#             print(flip[count])
#             count += 1
#             cv2.imwrite('tmp/{}.jpg'.format(count), img)
