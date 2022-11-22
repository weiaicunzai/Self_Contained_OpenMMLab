# import copy
import warnings
from collections.abc import Sequence
import collections
import warnings



import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from mmcv.parallel import DataContainer as DC
from numpy import random
import torch

# from ..builder import PIPELINES


# @PIPELINES.register_module()
#class ResizeToMultiple(object):
#    """Resize images & seg to multiple of divisor.
#    Args:
#        size_divisor (int): images and gt seg maps need to resize to multiple
#            of size_divisor. Default: 32.
#        interpolation (str, optional): The interpolation mode of image resize.
#            Default: None
#    """
#
#    def __init__(self, size_divisor=32, interpolation=None):
#        self.size_divisor = size_divisor
#        self.interpolation = interpolation
#
#    def __call__(self, results):
#        """Call function to resize images, semantic segmentation map to
#        multiple of size divisor.
#        Args:
#            results (dict): Result dict from loading pipeline.
#        Returns:
#            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
#        """
#        # Align image to multiple of size divisor.
#        img = results['img']
#        img = mmcv.imresize_to_multiple(
#            img,
#            self.size_divisor,
#            scale_factor=1,
#            interpolation=self.interpolation
#            if self.interpolation else 'bilinear')
#
#        results['img'] = img
#        results['img_shape'] = img.shape
#        results['pad_shape'] = img.shape
#
#        # Align segmentation map to multiple of size divisor.
#        for key in results.get('seg_fields', []):
#            gt_seg = results[key]
#            gt_seg = mmcv.imresize_to_multiple(
#                gt_seg,
#                self.size_divisor,
#                scale_factor=1,
#                interpolation='nearest')
#            results[key] = gt_seg
#
#        return results
#
#    def __repr__(self):
#        repr_str = self.__class__.__name__
#        repr_str += (f'(size_divisor={self.size_divisor}, '
#                     f'interpolation={self.interpolation})')
#        return repr_str


# @PIPELINES.register_module()
class Resize(object):
    """Resize images & seg.
    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.
    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:
    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)
    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)
    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
        min_size (int, optional): The minimum size for input and the shape
            of the image and seg map will not be less than ``min_size``.
            As the shape of model input is fixed like 'SETR' and 'BEiT'.
            Following the setting in these models, resized images must be
            bigger than the crop size in ``slide_inference``. Default: None
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 min_size=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.min_size = min_size

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        #results['scale'] = scale
        #results['scale_idx'] = scale_idx
        return scale, scale_idx

    def _resize_img(self, img, scale):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            if self.min_size is not None:
                # TODO: Now 'min_size' is an 'int' which means the minimum
                # shape of images is (min_size, min_size, 3). 'min_size'
                # with tuple type will be supported, i.e. the width and
                # height are not equal.
                #if min(results['scale']) < self.min_size:
                if min(scale) < self.min_size:
                    new_short = self.min_size
                else:
                    #new_short = min(results['scale'])
                    new_short = min(scale)

                #h, w = results['img'].shape[:2]
                h, w = img.shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                #results['scale'] = (new_h, new_w)
                scale = (new_h, new_w)

            # print(img.shape)
            img, scale_factor = mmcv.imrescale(
                img, scale, return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            #h, w = results['img'].shape[:2]
            h, w = img.shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
        #scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                #dtype=np.float32)
        # results['img'] = img
        # results['img_shape'] = img.shape
        # results['pad_shape'] = img.shape  # in case that there is no padding
        # results['scale_factor'] = scale_factor
        # results['keep_ratio'] = self.keep_ratio
        #return img, scale_factor
        return img

    #def _resize_seg(self, results):
    def _resize_seg(self, gt_seg, scale):
        """Resize semantic segmentation map with ``results['scale']``."""
        # for key in results.get('seg_fields', []):
        if self.keep_ratio:
            gt_seg = mmcv.imrescale(
                gt_seg, scale, interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                gt_seg, scale, interpolation='nearest')
            #results[key] = gt_seg
        return gt_seg

    def __call__(self, img, gt_seg, **kwargs):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        # if 'scale' not in results:
        print(kwargs)
        if 'scale' not in kwargs:
            print('hhhhhh')
            scale, scale_idx = self._random_scale(img)
        else:
            scale = kwargs['scale']

        img = self._resize_img(img, scale)
        gt_seg = self._resize_seg(gt_seg, scale)
        # self._resize_seg(results)
        return img, gt_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


#@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & seg.
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        #if 'flip' not in results:
            # flip = True if np.random.rand() < self.prob else False
            # results['flip'] = flip
        # if 'flip_direction' not in results:
            # results['flip_direction'] = self.direction

        if 'flip' not in kwargs:
            flip = True if np.random.rand() < self.prob else False
        else:
            flip = kwargs['flip']
        # if results['flip']:
        if 'flip_direction' not in kwargs:
            flip_direction = self.direction
        else:
            flip_direction = kwargs['flip_direction']

        if flip:
            # flip image
            #results['img'] = mmcv.imflip(
            #    results['img'], direction=results['flip_direction'])
            #img = mmcv.imflip(img, direction=self.direction)
            img = mmcv.imflip(img, direction=flip_direction)

            # flip segs
            #for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                #results[key] = mmcv.imflip(
                    #results[key], direction=results['flip_direction']).copy()

            gt_seg = mmcv.imflip(gt_seg, direction=self.direction).copy()
        return img, gt_seg

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


#@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    #def _pad_img(self, results):
    def _pad_img(self, img):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                # results['img'], shape=self.size, pad_val=self.pad_val)
                img, shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                # results['img'], self.size_divisor, pad_val=self.pad_val)
                img, self.size_divisor, pad_val=self.pad_val)
        # results['img'] = padded_img
        # results['pad_shape'] = padded_img.shape
        # results['pad_fixed_size'] = self.size
        # results['pad_size_divisor'] = self.size_divisor
        return padded_img

    def _pad_seg(self, gt_seg):
        """Pad masks according to ``results['pad_shape']``."""
        # for key in results.get('seg_fields', []):
            # results[key] = mmcv.impad(
        gt_seg = mmcv.impad(
                # results[key],
                gt_seg,
                shape=self.size,
                pad_val=self.seg_pad_val)

        return gt_seg

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        img = self._pad_img(img)
        gt_seg = self._pad_seg(gt_seg)
        return img, gt_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


#@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        #results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
        #                                  self.to_rgb)
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        #results['img_norm_cfg'] = dict(
        #    mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        #return results
        return img, gt_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


#@PIPELINES.register_module()
#class Rerange(object):
#    """Rerange the image pixel value.
#    Args:
#        min_value (float or int): Minimum value of the reranged image.
#            Default: 0.
#        max_value (float or int): Maximum value of the reranged image.
#            Default: 255.
#    """
#
#    def __init__(self, min_value=0, max_value=255):
#        assert isinstance(min_value, float) or isinstance(min_value, int)
#        assert isinstance(max_value, float) or isinstance(max_value, int)
#        assert min_value < max_value
#        self.min_value = min_value
#        self.max_value = max_value
#
#    def __call__(self, results):
#        """Call function to rerange images.
#        Args:
#            results (dict): Result dict from loading pipeline.
#        Returns:
#            dict: Reranged results.
#        """
#
#        img = results['img']
#        img_min_value = np.min(img)
#        img_max_value = np.max(img)
#
#        assert img_min_value < img_max_value
#        # rerange to [0, 1]
#        img = (img - img_min_value) / (img_max_value - img_min_value)
#        # rerange to [min_value, max_value]
#        img = img * (self.max_value - self.min_value) + self.min_value
#        results['img'] = img
#
#        return results
#
#    def __repr__(self):
#        repr_str = self.__class__.__name__
#        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
#        return repr_str
#
#
#@PIPELINES.register_module()
#class CLAHE(object):
#    """Use CLAHE method to process the image.
#    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
#    Graphics Gems, 1994:474-485.` for more information.
#    Args:
#        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
#        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
#            Input image will be divided into equally sized rectangular tiles.
#            It defines the number of tiles in row and column. Default: (8, 8).
#    """
#
#    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
#        assert isinstance(clip_limit, (float, int))
#        self.clip_limit = clip_limit
#        assert is_tuple_of(tile_grid_size, int)
#        assert len(tile_grid_size) == 2
#        self.tile_grid_size = tile_grid_size
#
#    def __call__(self, results):
#        """Call function to Use CLAHE method process images.
#        Args:
#            results (dict): Result dict from loading pipeline.
#        Returns:
#            dict: Processed results.
#        """
#
#        for i in range(results['img'].shape[2]):
#            results['img'][:, :, i] = mmcv.clahe(
#                np.array(results['img'][:, :, i], dtype=np.uint8),
#                self.clip_limit, self.tile_grid_size)
#
#        return results
#
#    def __repr__(self):
#        repr_str = self.__class__.__name__
#        repr_str += f'(clip_limit={self.clip_limit}, '\
#                    f'tile_grid_size={self.tile_grid_size})'
#        return repr_str
#
#
#@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & seg.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        #img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                #seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                seg_temp = self.crop(gt_seg, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        # img_shape = img.shape
        # results['img'] = img
        # results['img_shape'] = img_shape

        # crop semantic seg
        # for key in results.get('seg_fields', []):
            # results[key] = self.crop(results[key], crop_bbox)

        gt_seg = self.crop(gt_seg, crop_bbox)

        return img, gt_seg

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


#@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate the image & seg.
    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to rotate image, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            #results['img'] = mmcv.imrotate(
            img = mmcv.imrotate(
                #results['img'],
                img,
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            #for key in results.get('seg_fields', []):
            gt_seg = mmcv.imrotate(
                    #results[key],
                    gt_seg,
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        #return results
        return img, gt_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


#@PIPELINES.register_module()
#class RGB2Gray(object):
#    """Convert RGB image to grayscale image.
#    This transform calculate the weighted mean of input image channels with
#    ``weights`` and then expand the channels to ``out_channels``. When
#    ``out_channels`` is None, the number of output channels is the same as
#    input channels.
#    Args:
#        out_channels (int): Expected number of output channels after
#            transforming. Default: None.
#        weights (tuple[float]): The weights to calculate the weighted mean.
#            Default: (0.299, 0.587, 0.114).
#    """
#
#    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
#        assert out_channels is None or out_channels > 0
#        self.out_channels = out_channels
#        assert isinstance(weights, tuple)
#        for item in weights:
#            assert isinstance(item, (float, int))
#        self.weights = weights
#
#    def __call__(self, results):
#        """Call function to convert RGB image to grayscale image.
#        Args:
#            results (dict): Result dict from loading pipeline.
#        Returns:
#            dict: Result dict with grayscale image.
#        """
#        img = results['img']
#        assert len(img.shape) == 3
#        assert img.shape[2] == len(self.weights)
#        weights = np.array(self.weights).reshape((1, 1, -1))
#        img = (img * weights).sum(2, keepdims=True)
#        if self.out_channels is None:
#            img = img.repeat(weights.shape[2], axis=2)
#        else:
#            img = img.repeat(self.out_channels, axis=2)
#
#        results['img'] = img
#        results['img_shape'] = img.shape
#
#        return results
#
#    def __repr__(self):
#        repr_str = self.__class__.__name__
#        repr_str += f'(out_channels={self.out_channels}, ' \
#                    f'weights={self.weights})'
#        return repr_str
#
#
#@PIPELINES.register_module()
#class AdjustGamma(object):
#    """Using gamma correction to process the image.
#    Args:
#        gamma (float or int): Gamma value used in gamma correction.
#            Default: 1.0.
#    """
#
#    def __init__(self, gamma=1.0):
#        assert isinstance(gamma, float) or isinstance(gamma, int)
#        assert gamma > 0
#        self.gamma = gamma
#        inv_gamma = 1.0 / gamma
#        self.table = np.array([(i / 255.0)**inv_gamma * 255
#                               for i in np.arange(256)]).astype('uint8')
#
#    def __call__(self, results):
#        """Call function to process the image with gamma correction.
#        Args:
#            results (dict): Result dict from loading pipeline.
#        Returns:
#            dict: Processed results.
#        """
#
#        results['img'] = mmcv.lut_transform(
#            np.array(results['img'], dtype=np.uint8), self.table)
#
#        return results
#
#    def __repr__(self):
#        return self.__class__.__name__ + f'(gamma={self.gamma})'
#
#
#@PIPELINES.register_module()
#class SegRescale(object):
#    """Rescale semantic segmentation maps.
#    Args:
#        scale_factor (float): The scale factor of the final output.
#    """
#
#    def __init__(self, scale_factor=1):
#        self.scale_factor = scale_factor
#
#    def __call__(self, results):
#        """Call function to scale the semantic segmentation map.
#        Args:
#            results (dict): Result dict from loading pipeline.
#        Returns:
#            dict: Result dict with semantic segmentation map scaled.
#        """
#        for key in results.get('seg_fields', []):
#            if self.scale_factor != 1:
#                results[key] = mmcv.imrescale(
#                    results[key], self.scale_factor, interpolation='nearest')
#        return results
#
#    def __repr__(self):
#        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'
#
#
#@PIPELINES.register_module()
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
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
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
        mode = random.randint(2)
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


#@PIPELINES.register_module()
#class RandomCutOut(object):
#    """CutOut operation.
#    Randomly drop some regions of image used in
#    `Cutout <https://arxiv.org/abs/1708.04552>`_.
#    Args:
#        prob (float): cutout probability.
#        n_holes (int | tuple[int, int]): Number of regions to be dropped.
#            If it is given as a list, number of holes will be randomly
#            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
#        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
#            shape of dropped regions. It can be `tuple[int, int]` to use a
#            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
#            shape from the list.
#        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
#            candidate ratio of dropped regions. It can be `tuple[float, float]`
#            to use a fixed ratio or `list[tuple[float, float]]` to randomly
#            choose ratio from the list. Please note that `cutout_shape`
#            and `cutout_ratio` cannot be both given at the same time.
#        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
#            of pixel to fill in the dropped regions. Default: (0, 0, 0).
#        seg_fill_in (int): The labels of pixel to fill in the dropped regions.
#            If seg_fill_in is None, skip. Default: None.
#    """
#
#    def __init__(self,
#                 prob,
#                 n_holes,
#                 cutout_shape=None,
#                 cutout_ratio=None,
#                 fill_in=(0, 0, 0),
#                 seg_fill_in=None):
#
#        assert 0 <= prob and prob <= 1
#        assert (cutout_shape is None) ^ (cutout_ratio is None), \
#            'Either cutout_shape or cutout_ratio should be specified.'
#        assert (isinstance(cutout_shape, (list, tuple))
#                or isinstance(cutout_ratio, (list, tuple)))
#        if isinstance(n_holes, tuple):
#            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
#        else:
#            n_holes = (n_holes, n_holes)
#        if seg_fill_in is not None:
#            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
#                    and seg_fill_in <= 255)
#        self.prob = prob
#        self.n_holes = n_holes
#        self.fill_in = fill_in
#        self.seg_fill_in = seg_fill_in
#        self.with_ratio = cutout_ratio is not None
#        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
#        if not isinstance(self.candidates, list):
#            self.candidates = [self.candidates]
#
#    def __call__(self, results):
#        """Call function to drop some regions of image."""
#        cutout = True if np.random.rand() < self.prob else False
#        if cutout:
#            h, w, c = results['img'].shape
#            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
#            for _ in range(n_holes):
#                x1 = np.random.randint(0, w)
#                y1 = np.random.randint(0, h)
#                index = np.random.randint(0, len(self.candidates))
#                if not self.with_ratio:
#                    cutout_w, cutout_h = self.candidates[index]
#                else:
#                    cutout_w = int(self.candidates[index][0] * w)
#                    cutout_h = int(self.candidates[index][1] * h)
#
#                x2 = np.clip(x1 + cutout_w, 0, w)
#                y2 = np.clip(y1 + cutout_h, 0, h)
#                results['img'][y1:y2, x1:x2, :] = self.fill_in
#
#                if self.seg_fill_in is not None:
#                    for key in results.get('seg_fields', []):
#                        results[key][y1:y2, x1:x2] = self.seg_fill_in
#
#        return results
#
#    def __repr__(self):
#        repr_str = self.__class__.__name__
#        repr_str += f'(prob={self.prob}, '
#        repr_str += f'n_holes={self.n_holes}, '
#        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
#                     else f'cutout_shape={self.candidates}, ')
#        repr_str += f'fill_in={self.fill_in}, '
#        repr_str += f'seg_fill_in={self.seg_fill_in})'
#        return repr_str
#
#
#@PIPELINES.register_module()
#class RandomMosaic(object):
#    """Mosaic augmentation. Given 4 images, mosaic transform combines them into
#    one output image. The output image is composed of the parts from each sub-
#    image.
#    .. code:: text
#                        mosaic transform
#                           center_x
#                +------------------------------+
#                |       pad        |  pad      |
#                |      +-----------+           |
#                |      |           |           |
#                |      |  image1   |--------+  |
#                |      |           |        |  |
#                |      |           | image2 |  |
#     center_y   |----+-------------+-----------|
#                |    |   cropped   |           |
#                |pad |   image3    |  image4   |
#                |    |             |           |
#                +----|-------------+-----------+
#                     |             |
#                     +-------------+
#     The mosaic transform steps are as follows:
#         1. Choose the mosaic center as the intersections of 4 images
#         2. Get the left top image according to the index, and randomly
#            sample another 3 images from the custom dataset.
#         3. Sub image will be cropped if image is larger than mosaic patch
#    Args:
#        prob (float): mosaic probability.
#        img_scale (Sequence[int]): Image size after mosaic pipeline of
#            a single image. The size of the output image is four times
#            that of a single image. The output image comprises 4 single images.
#            Default: (640, 640).
#        center_ratio_range (Sequence[float]): Center ratio range of mosaic
#            output. Default: (0.5, 1.5).
#        pad_val (int): Pad value. Default: 0.
#        seg_pad_val (int): Pad value of segmentation map. Default: 255.
#    """
#
#    def __init__(self,
#                 prob,
#                 img_scale=(640, 640),
#                 center_ratio_range=(0.5, 1.5),
#                 pad_val=0,
#                 seg_pad_val=255):
#        assert 0 <= prob and prob <= 1
#        assert isinstance(img_scale, tuple)
#        self.prob = prob
#        self.img_scale = img_scale
#        self.center_ratio_range = center_ratio_range
#        self.pad_val = pad_val
#        self.seg_pad_val = seg_pad_val
#
#    def __call__(self, results):
#        """Call function to make a mosaic of image.
#        Args:
#            results (dict): Result dict.
#        Returns:
#            dict: Result dict with mosaic transformed.
#        """
#        mosaic = True if np.random.rand() < self.prob else False
#        if mosaic:
#            results = self._mosaic_transform_img(results)
#            results = self._mosaic_transform_seg(results)
#        return results
#
#    def get_indexes(self, dataset):
#        """Call function to collect indexes.
#        Args:
#            dataset (:obj:`MultiImageMixDataset`): The dataset.
#        Returns:
#            list: indexes.
#        """
#
#        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
#        return indexes
#
#    def _mosaic_transform_img(self, results):
#        """Mosaic transform function.
#        Args:
#            results (dict): Result dict.
#        Returns:
#            dict: Updated result dict.
#        """
#
#        assert 'mix_results' in results
#        if len(results['img'].shape) == 3:
#            mosaic_img = np.full(
#                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
#                self.pad_val,
#                dtype=results['img'].dtype)
#        else:
#            mosaic_img = np.full(
#                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
#                self.pad_val,
#                dtype=results['img'].dtype)
#
#        # mosaic center x, y
#        self.center_x = int(
#            random.uniform(*self.center_ratio_range) * self.img_scale[1])
#        self.center_y = int(
#            random.uniform(*self.center_ratio_range) * self.img_scale[0])
#        center_position = (self.center_x, self.center_y)
#
#        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
#        for i, loc in enumerate(loc_strs):
#            if loc == 'top_left':
#                result_patch = copy.deepcopy(results)
#            else:
#                result_patch = copy.deepcopy(results['mix_results'][i - 1])
#
#            img_i = result_patch['img']
#            h_i, w_i = img_i.shape[:2]
#            # keep_ratio resize
#            scale_ratio_i = min(self.img_scale[0] / h_i,
#                                self.img_scale[1] / w_i)
#            img_i = mmcv.imresize(
#                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
#
#            # compute the combine parameters
#            paste_coord, crop_coord = self._mosaic_combine(
#                loc, center_position, img_i.shape[:2][::-1])
#            x1_p, y1_p, x2_p, y2_p = paste_coord
#            x1_c, y1_c, x2_c, y2_c = crop_coord
#
#            # crop and paste image
#            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
#
#        results['img'] = mosaic_img
#        results['img_shape'] = mosaic_img.shape
#        results['ori_shape'] = mosaic_img.shape
#
#        return results
#
#    def _mosaic_transform_seg(self, results):
#        """Mosaic transform function for label annotations.
#        Args:
#            results (dict): Result dict.
#        Returns:
#            dict: Updated result dict.
#        """
#
#        assert 'mix_results' in results
#        for key in results.get('seg_fields', []):
#            mosaic_seg = np.full(
#                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
#                self.seg_pad_val,
#                dtype=results[key].dtype)
#
#            # mosaic center x, y
#            center_position = (self.center_x, self.center_y)
#
#            loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
#            for i, loc in enumerate(loc_strs):
#                if loc == 'top_left':
#                    result_patch = copy.deepcopy(results)
#                else:
#                    result_patch = copy.deepcopy(results['mix_results'][i - 1])
#
#                gt_seg_i = result_patch[key]
#                h_i, w_i = gt_seg_i.shape[:2]
#                # keep_ratio resize
#                scale_ratio_i = min(self.img_scale[0] / h_i,
#                                    self.img_scale[1] / w_i)
#                gt_seg_i = mmcv.imresize(
#                    gt_seg_i,
#                    (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
#                    interpolation='nearest')
#
#                # compute the combine parameters
#                paste_coord, crop_coord = self._mosaic_combine(
#                    loc, center_position, gt_seg_i.shape[:2][::-1])
#                x1_p, y1_p, x2_p, y2_p = paste_coord
#                x1_c, y1_c, x2_c, y2_c = crop_coord
#
#                # crop and paste image
#                mosaic_seg[y1_p:y2_p, x1_p:x2_p] = gt_seg_i[y1_c:y2_c,
#                                                            x1_c:x2_c]
#
#            results[key] = mosaic_seg
#
#        return results
#
#    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
#        """Calculate global coordinate of mosaic image and local coordinate of
#        cropped sub-image.
#        Args:
#            loc (str): Index for the sub-image, loc in ('top_left',
#              'top_right', 'bottom_left', 'bottom_right').
#            center_position_xy (Sequence[float]): Mixing center for 4 images,
#                (x, y).
#            img_shape_wh (Sequence[int]): Width and height of sub-image
#        Returns:
#            tuple[tuple[float]]: Corresponding coordinate of pasting and
#                cropping
#                - paste_coord (tuple): paste corner coordinate in mosaic image.
#                - crop_coord (tuple): crop corner coordinate in mosaic image.
#        """
#
#        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
#        if loc == 'top_left':
#            # index0 to top left part of image
#            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
#                             max(center_position_xy[1] - img_shape_wh[1], 0), \
#                             center_position_xy[0], \
#                             center_position_xy[1]
#            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
#                y2 - y1), img_shape_wh[0], img_shape_wh[1]
#
#        elif loc == 'top_right':
#            # index1 to top right part of image
#            x1, y1, x2, y2 = center_position_xy[0], \
#                             max(center_position_xy[1] - img_shape_wh[1], 0), \
#                             min(center_position_xy[0] + img_shape_wh[0],
#                                 self.img_scale[1] * 2), \
#                             center_position_xy[1]
#            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
#                img_shape_wh[0], x2 - x1), img_shape_wh[1]
#
#        elif loc == 'bottom_left':
#            # index2 to bottom left part of image
#            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
#                             center_position_xy[1], \
#                             center_position_xy[0], \
#                             min(self.img_scale[0] * 2, center_position_xy[1] +
#                                 img_shape_wh[1])
#            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
#                y2 - y1, img_shape_wh[1])
#
#        else:
#            # index3 to bottom right part of image
#            x1, y1, x2, y2 = center_position_xy[0], \
#                             center_position_xy[1], \
#                             min(center_position_xy[0] + img_shape_wh[0],
#                                 self.img_scale[1] * 2), \
#                             min(self.img_scale[0] * 2, center_position_xy[1] +
#                                 img_shape_wh[1])
#            crop_coord = 0, 0, min(img_shape_wh[0],
#                                   x2 - x1), min(y2 - y1, img_shape_wh[1])
#
#        paste_coord = x1, y1, x2, y2
#        return paste_coord, crop_coord
#
#    def __repr__(self):
#        repr_str = self.__class__.__name__
#        repr_str += f'(prob={self.prob}, '
#        repr_str += f'img_scale={self.img_scale}, '
#        repr_str += f'center_ratio_range={self.center_ratio_range}, '
#        repr_str += f'pad_val={self.pad_val}, '
#        repr_str += f'seg_pad_val={self.pad_val})'
#        return repr_str


class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.
    An example configuration is as followed:
    .. code-block::
        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:
    .. code-block::
        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )
    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        if flip:
            trans_index = {
                key['type']: index
                for index, key in enumerate(transforms)
            }
            if 'RandomFlip' in trans_index and 'Pad' in trans_index:
                assert trans_index['RandomFlip'] < trans_index['Pad'], \
                    'Pad must be executed after RandomFlip when flip is True'
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert mmcv.is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(
                img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    # def __call__(self, results):
    def __call__(self, img, gt_seg):
        """Call function to apply test time augment transforms on results.
        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            # h, w = results['img'].shape[:2]
            h, w = img.shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    # _results = results.copy()
                    # _results['scale'] = scale
                    # _results['flip'] = flip
                    # _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str



def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


#@PIPELINES.register_module()
#class ToTensor(object):
#    """Convert some results to :obj:`torch.Tensor` by given keys.
#    Args:
#        keys (Sequence[str]): Keys that need to be converted to Tensor.
#    """
#
#    def __init__(self, keys):
#        self.keys = keys
#
#    def __call__(self, results):
#        """Call function to convert data in results to :obj:`torch.Tensor`.
#        Args:
#            results (dict): Result dict contains the data to convert.
#        Returns:
#            dict: The result dict contains the data converted
#                to :obj:`torch.Tensor`.
#        """
#
#        for key in self.keys:
#            results[key] = to_tensor(results[key])
#        return results
#
#    def __repr__(self):
#        return self.__class__.__name__ + f'(keys={self.keys})'
#
#
#@PIPELINES.register_module()
#class ImageToTensor(object):
#    """Convert image to :obj:`torch.Tensor` by given keys.
#    The dimension order of input image is (H, W, C). The pipeline will convert
#    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
#    (1, H, W).
#    Args:
#        keys (Sequence[str]): Key of images to be converted to Tensor.
#    """
#
#    def __init__(self, keys):
#        self.keys = keys
#
#    def __call__(self, results):
#        """Call function to convert image in results to :obj:`torch.Tensor` and
#        transpose the channel order.
#        Args:
#            results (dict): Result dict contains the image data to convert.
#        Returns:
#            dict: The result dict contains the image converted
#                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
#        """
#
#        for key in self.keys:
#            img = results[key]
#            if len(img.shape) < 3:
#                img = np.expand_dims(img, -1)
#            results[key] = to_tensor(img.transpose(2, 0, 1))
#        return results
#
#    def __repr__(self):
#        return self.__class__.__name__ + f'(keys={self.keys})'
#
#
#@PIPELINES.register_module()
#class Transpose(object):
#    """Transpose some results by given keys.
#    Args:
#        keys (Sequence[str]): Keys of results to be transposed.
#        order (Sequence[int]): Order of transpose.
#    """
#
#    def __init__(self, keys, order):
#        self.keys = keys
#        self.order = order
#
#    def __call__(self, results):
#        """Call function to convert image in results to :obj:`torch.Tensor` and
#        transpose the channel order.
#        Args:
#            results (dict): Result dict contains the image data to convert.
#        Returns:
#            dict: The result dict contains the image converted
#                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
#        """
#
#        for key in self.keys:
#            results[key] = results[key].transpose(self.order)
#        return results
#
#    def __repr__(self):
#        return self.__class__.__name__ + \
#               f'(keys={self.keys}, order={self.order})'
#
#
#@PIPELINES.register_module()
#class ToDataContainer(object):
#    """Convert results to :obj:`mmcv.DataContainer` by given fields.
#    Args:
#        fields (Sequence[dict]): Each field is a dict like
#            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
#            be converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
#            Default: ``(dict(key='img', stack=True),
#            dict(key='gt_semantic_seg'))``.
#    """
#
#    def __init__(self,
#                 fields=(dict(key='img',
#                              stack=True), dict(key='gt_semantic_seg'))):
#        self.fields = fields
#
#    def __call__(self, results):
#        """Call function to convert data in results to
#        :obj:`mmcv.DataContainer`.
#        Args:
#            results (dict): Result dict contains the data to convert.
#        Returns:
#            dict: The result dict contains the data converted to
#                :obj:`mmcv.DataContainer`.
#        """
#
#        for field in self.fields:
#            field = field.copy()
#            key = field.pop('key')
#            results[key] = DC(results[key], **field)
#        return results
#
#    def __repr__(self):
#        return self.__class__.__name__ + f'(fields={self.fields})'
#
#
#@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        # if 'img' in results:
            # img = results['img']
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        # results['img'] = DC(to_tensor(img), stack=True)
        img = to_tensor(img)
        #if 'gt_semantic_seg' in results:
        #    # convert to long
        #    results['gt_semantic_seg'] = DC(
        #        to_tensor(results['gt_semantic_seg'][None,
        #                                             ...].astype(np.int64)),
        #        stack=True)
        gt_seg = to_tensor(gt_seg.astype(np.int64))
        #return results
        return img, gt_seg

    def __repr__(self):
        return self.__class__.__name__

class Compose(object):
    """Compose multiple transforms sequentially.
    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            # if isinstance(transform, dict):
                # transform = build_from_cfg(transform, PIPELINES)
                # self.transforms.append(transform)
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable')

    # def __call__(self, data):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to apply transforms sequentially.
        Args:
            data (dict): A result dict contains the data to transform.
        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            # data = t(data)
            img, gt_seg = t(img, gt_seg, **kwargs)
            # if data is None:
                # return None
        # return data
        return img, gt_seg

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string



class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.
    An example configuration is as followed:
    .. code-block::
        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:
    .. code-block::
        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )
    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        if flip:
            trans_index = {
                #key['type']: index
                key.__class__.__name__: index
                for index, key in enumerate(transforms)
            }
            if 'RandomFlip' in trans_index and 'Pad' in trans_index:
                assert trans_index['RandomFlip'] < trans_index['Pad'], \
                    'Pad must be executed after RandomFlip when flip is True'
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert mmcv.is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(
                img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                #and not any([t['type'] == 'RandomFlip' for t in transforms])):
                and not any([t.__class__.__name__ == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    #def __call__(self, results):
    def __call__(self, img, gt_seg, **kwargs):
        """Call function to apply test time augment transforms on results.
        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            #h, w = results['img'].shape[:2]
            h, w = img.shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    # _results = results.copy()
                    # _results['scale'] = scale
                    # _results['flip'] = flip
                    # _results['flip_direction'] = direction
                    # print(self.transforms, '.llll')
                    data = self.transforms(
                        img, 
                        gt_seg, 
                        scale=scale, 
                        flip=flip, 
                        flip_direction=direction
                    )
                    aug_data.append(data)
        # list of dict to dict of list
        print(len(aug_data))
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str


#trans = Resize(img_scale=(400, 400), ratio_range=(0.5, 2.0))
#trans = RandomCrop(crop_size=(500, 500), cat_max_ratio=0.75)
#trans = RandomFlip(prob=0.5)
#trans = PhotoMetricDistortion()
#trans = Normalize(mean=(1, 1, 1), std=(1,1,1))
#trans = DefaultFormatBundle()
#trans = RandomRotate(prob=1, degree=(0, 45))
#trans = Pad(size=(500, 500))
##trans = Resize(img_scale=(1000, 500))
#
# seg = np.arange(500 * 500 * 3).reshape(500, 500, 3).astype('uint8')
#
my_trans = Compose(my_trans)



import cv2
image = cv2.imread('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/mmseg_glas/train/train_10.bmp')
# image = image[:500, :500]
seg = np.arange(image.shape[0] *  image.shape[1] * 3).reshape(image.shape[0], image.shape[1], 3).astype('uint8')
# image, seg = trans(image, seg)
image, seg = my_trans(image, seg)

print(image.shape, seg.shape)
#    

cv2.imwrite('ff.jpg', image)
#image, seg = trans(image, image)
#print(image.shape, seg.shape)
#    

#trans = MultiScaleFlipAug(
#    flip=True,
#    img_scale=(400, 400),
#    img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
#    transforms=[Resize(keep_ratio=True),RandomFlip()]
#    )
#
#image = np.arange(500 * 500 * 3).reshape(500, 500, 3).astype('uint8')
#image, seg = trans(image, image)
#print(image.shape, seg.shape)


cv2.imwrite('ff.jpg', image)