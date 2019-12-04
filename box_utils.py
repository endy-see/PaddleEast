from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format"""
    if isinstance(xywh, (list, tuple)):
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2]-1.)
        y2 = y1 + np.maximum(0., xywh[3]-1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        return np.hstack((xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1)))
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width"""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2
