from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from edict import AttrDict
import six
import numpy as np

_C = AttrDict()
cfg = _C

"""
Training options
"""
_C.TRAIN = AttrDict()

# scales an image's shortest side
_C.TRAIN.scales = [800]

# max size of longest side
_C.TRAIN.max_size = 1333

# images per GPU in minibatch
_C.TRAIN.im_per_batch = 1

# stopgrad at a spcified stage
_C.TRAIN.freeze_at = 2

# number of RPN proposals to keep before NMS
_C.TRAIN.rpn_pre_nms_top_n = 12000

# number of RPN proposals to keep after NMS
_C.TRAIN.rpn_post_nms_top_n = 2000

# NMS threshold used on RPN proposals
_C.TRAIN.rpn_nms_thresh = 0.7

# min size in RPN proposals
_C.TRAIN.rpn_min_size = 0

# eta for adaptive NMS in RPN
_C.TRAIN.rpn_eta = 1.0

# roi minibatch size per image
_C.TRAIN.batch_size_per_im = 512

# target fraction of foreground roi minibatch
_C.TRAIN.fg_fraction = 0.25

# overlap threshold for a foreground roi
_C.TRAIN.fg_thresh = 0.5

# overlap threshold for a background roi
_C.TRAIN.bg_thresh_hi = 0.5
_C.TRAIN.bg_thresh_lo = 0.0

# weight for bbox resgression targets
_C.bbox_reg_weights = [0.1, 0.1, 0.2, 0.2]

# Class number
_C.class_num = 81

# min area of ground truth box
_C.TRAIN.gt_min_area = 0

# If False, only resize image and not pad, image shape is different between GPUs in one mini-batch.
# If True, image shape is the same in one mini-batch.
_C.TRAIN.padding_minibatch = False

# number of RPN examples per image
_C.TRAIN.rpn_batch_size_per_im = 256

# remove anchors out of the image
_C.TRAIN.rpn_straddle_thresh = 0.

# target fraction of foreground examples per RPN minibatch
_C.TRAIN.rpn_fg_fraction = 0.5

# min overlap between anchor and gt box to be a positive examples
_C.TRAIN.rpn_positive_overlap = 0.7

# max overlap between anchor and gt box to be a negative examples
_C.TRAIN.rpn_negative_overlap = 0.3

# use horizontally-flipped images during training?
_C.TRAIN.use_flipped = True

# Snapshot period
_C.TRAIN.snapshot_iter = 10000

"""
Inference options
"""
_C.TEST = AttrDict()

# scales an image's shortest side
_C.TEST.scales = [800]

# eta for adaptive NMS in RPN
_C.TEST.rpn_eta = 1.0

# max size of longest side
_C.TEST.max_size = 1333

# min score threshold used for NMS
_C.TEST.score_thresh = 0.05

# overlap threshold used for NMS
_C.TEST.nms_thresh = 0.5

# number of RPN proposals to keep before NMS
_C.TEST.rpn_pre_nms_top_n = 6000

# number of RPN proposals to keep after NMS
_C.TEST.rpn_pos_nms_top_n = 1000

# min size in RPN proposals
_C.TEST.rpn_min_size = 0.0

# max number of detections
_C.TEST.detections_per_im = 100

# NMS threshold used on RPN proposals
_C.TEST.rpn_nms_thresh = 0.7

"""
Model options
"""
# weight for bbox regression targets
_C.bbox_reg_weights = [0.1, 0.1, 0.2, 0.2]

# RPN anchor sizes
_C.anchor_sizes = [32, 64, 128, 256, 512]

# RPN anchor ratio
_C.aspect_ratio = [0.5, 1, 2]

# variance of anchors
_C.variances = [1., 1., 1., 1.]

# stride of feature map
_C.rpn_stride = [16.0, 16.0]

# Use roi pool or roi align, 'RoIPool' or 'RoIAlign'
_C.roi_func = 'RoIAlign'

# pooled width and pooled height
_C.roi_resolution = 14

# spatial scale
_C.spatial_scale = 1. / 16.

# sampling ratio for roi align
_C.sampling_ratio = 0

"""
SOLVER options
"""
# derived learning rate to get the final learning rate
_C.learning_rate = 0.01

# warm up to learning rate
_C.warm_up_iter = 500
_C.warm_up_factor = 1. / 3.

# lr steps with decay, 1x: [120000, 160000], 2x: [240000, 320000]
_C.lr_steps = [120000, 160000]
_C.lr_gamma = 0.1

# L2 regularization hyperparameter
_C.weight_decay = 0.0001

# momentum with SGD
_C.momentum = 0.9

# maximum number of iterations, 1x: 180000, 2x: 360000
_C.max_iter = 180000

"""
ENV options
"""
# support both CPU and GPU
_C.use_gpu = True

# whether use parallel
_C.parallel = True

# support pyreader
_C.use_pyreader = True

# pixel mean values
_C.pixel_means = [102.9801, 115.9465, 122.7717]


def merge_cfg_from_args(args, mode):
    """Merge config keys,values in args into the global config."""
    if mode == 'train':
        sub_d = _C.TRAIN
    else:
        sub_d = _C.TEST
    for k, v in sorted(six.iteritems(vars(args))):
        d = _C
        try:
            value = eval(v)
        except:
            value = v
        if k in sub_d:
            sub_d[k] = value
        else:
            d[k] = value


