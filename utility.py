"""
Contains common utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import distutils.util
import argparse
import functools
from config import *
import collections
from collections import deque
import datetime
import paddle.fluid as fluid
from logging import getLogger


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """
    Add argparse's argument.
    Usage:
        parser = argparse.ArgumentParser()
        add_arguments("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs
    )


def parse_args():
    """return all args"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # ENV
    add_arg('parallel',                   bool,             True,                           "Whether use parallel.")
    add_arg('use_gpu',                    bool,             True,                           "Whether use GPU.")
    add_arg('model_save_dir',             str,              'output',                       "The path to save model.")
    add_arg('pretrained_model',           str,              'res50',                        "The init model path.")
    add_arg('dataset',                    str,              'coco2017',                     "ICDAR2015.")
    add_arg('data_dir',                   str,              'dataset/icdar',                "The data root path.")
    add_arg('class_num',                  str,              '81',                           "Class number.")
    add_arg('use_pyreader',               bool,             True,                           "Use pyreader.")
    add_arg('padding_minibatch',          bool,             False,                          "If False, only resize image and not pad,image shape is different between GPUs in one mini-batch. If True, image shape is the same in one mini-batch.")
    #SOLVER
    add_arg('learning_rate',              float,            0.01,                           "Learning rate(default 8 GPUs.)")
    add_arg('max_iter',                   int,              180000,                         "Iter number.")
    add_arg('log_window',                 int,              20,                             "Log smooth window, set 1 for debug, set 20 for train.")
    # EAST todo
    # TRAIN VAL INFER
    add_arg('im_per_batch',               int,              1,                              "Minibatch size.")
    add_arg('max_size',                   int,              1333,                           "The resized image max height.")
    add_arg('scale',                      int,              [800],                          "The resized image height.")
    add_arg('batch_size_per_im',          int,              512,                            "East batch size.")
    add_arg('pixel_means',                float,            [102.9801, 115.9465, 122.7717], "pixel mean.")
    add_arg('nms_thresh',                 float,            0.5,                            "NMS threshold.")
    add_arg('score_thresh',               float,            0.05,                           "Score threshold for NMS.")
    add_arg('snapshot_stride',            int,              2000,                           "Save model every snapshot stride.")
    # SINGLE EVAL AND DRAW
    add_arg('draw_threshold',             float,            0.8,                            "Confidence threshold to draw bbox.")
    add_arg('image_path',                 str,              'dataset/icdar/val',            "The image path used to inference and visualize.")

    args = parser.parse_args()
    file_name = sys.argv[0]
    if 'train' in file_name:
        merge_cfg_from_args(args, 'train')
    else:
        merge_cfg_from_args(args, 'val')
    return args


def print_arguments(args):
    """
    Print argparse's arguments
    Usage:
        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="John", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s, %s" % (arg, value))
    print("------------------------------------------------")


class SmoothedValue(object):
    """
    Track a series of value and provide access to smoothed values over a window or the global series average.
    """
    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)

    def add_value(self, value):
        self.deque.append(value)

    def get_median_value(self):
        return np.median(self.deque)


class TrainingStats(object):
    def __init__(self, window_size, stats_keys):
        self.smoothed_losses_and_matrics = {
            key: SmoothedValue(window_size) for key in stats_keys
        }

    def update(self, stats):
        for k, v in self.smoothed_losses_and_matrics.items():
            v.add_value(stats[k])

    def get(self, extras=None):
        stats = collections.OrderedDict()
        if extras:
            for k, v in extras.items():
                stats[k] = v
        for k, v in self.smoothed_losses_and_matrics.items():
            stats[k] = round(v.get_median_value(), 3)
        return stats

    def log(self, extras=None):
        d = self.get(extras)
        strs = ', '.join(str(dict({x: y})).strip('{}') for x, y in d.items())
        return strs


def now_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle cpu version
    :param use_gpu:
    :return:
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"
    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            logger = getLogger('MachineErrorLog')
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass









