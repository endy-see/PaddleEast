import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from config import cfg
import math


def add_ResNet50_conv4_body(body_input):
    stages, block_func = ResNet_cfg[50]
    stages = stages[0:3]
    conv1 = conv_affine_layer(body_input, ch_out=64, filter_size=7, stride=2, padding=3, name="conv1")
    pool1 = fluid.layers.pool2d(input=conv1, pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)
    res2 = layer_warp(block_func, pool1, 64, stages[0], 1, name="res2")
    if cfg.TRAIN.freeze_at == 2:
        res2.stop_gradient = True
    res3 = layer_warp(block_func, res2, 128, stages[1], 2, name="res3")
    if cfg.TRAIN.freeze_at == 3:
        res3.stop_gradient = True
    res4 = layer_warp(block_func, res3, 256, stages[2], 2, name="res4")
    if cfg.TRAIN.freeze_at == 4:
        res4.stop_gradient = True
    return res4


def add_ResNet50_convs_body(body_input):
    conv_list = []
    stages, block_func = ResNet_cfg[50]
    stages = stages[0:3]
    conv1 = conv_affine_layer(body_input, ch_out=64, filter_size=7, stride=2, padding=3, name="conv1")
    pool1 = fluid.layers.pool2d(input=conv1, pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)
    res2 = layer_warp(block_func, pool1, 64, stages[0], 1, name="res2")
    conv_list.append(res2)
    res3 = layer_warp(block_func, res2, 128, stages[1], 2, name="res3")
    conv_list.append(res3)
    res4 = layer_warp(block_func, res3, 256, stages[2], 2, name="res4")
    conv_list.append(res4)
    res5 = layer_warp(block_func, res4, 512, stages[3], 2, name="res5")
    conv_list.append(res5)
    return conv_list


def upsample(input, scale=2, name=None):
    out = fluid.layers.resize_nearest(input=input, scale=float(scale), name=name)
    # out = fluid.layers.resize_bilinear(input=input, scale=float(scale), name=name)    Todo
    return out


def conv_bn_relu_layer(input, num_filters, filter_size, stride=1, padding=0, groups=1, act='relu', name=None):
    conv = fluid.layers.conv2d(input=input, num_filters=num_filters, filter_size=filter_size, stride=stride, padding=padding, groups=groups, act=None, bias_attr=False, name=name)
    return fluid.layers.batch_norm(input=conv, act=act)


def last_conv(input, num_filters, filter_size, stride=1, padding=0, groups=1,  name=None):
    return fluid.layers.conv2d(input=input, num_filters=num_filters, filter_size=filter_size, stride=stride, padding=padding, groups=groups, act=None, bias_attr=False, name=name)


def add_feature_merging_func(conv_list):
    """do feature merging, input should be resnet's every layer"""
    res5 = conv_list[3]     # bs 2048 w/32 h/32
    res4_unpool = upsample(res5)
    res4 = conv_list[2]
    concat1 = fluid.layers.concat([res4_unpool, res4], axis=1)
    concat1_conv1 = conv_bn_relu_layer(input=concat1, num_filters=128, filter_size=1, name='concat1_conv1')
    concat1_conv3 = conv_bn_relu_layer(input=concat1_conv1, num_filters=128, filter_size=3, padding=1, name='concat1_conv3')

    res3_unpool = upsample(concat1_conv3)
    res3 = conv_list[1]
    concat2 = fluid.layers.concat([res3_unpool, res3], axis=1)
    concat2_conv1 = conv_bn_relu_layer(input=concat2, num_filters=64, filter_size=1, name='concat2_conv2')
    concat2_conv3 = conv_bn_relu_layer(input=concat2_conv1, num_filters=64, filter_size=3, padding=1, name='concat2_conv3')

    res2_unpool = upsample(concat2_conv3)
    res2 = conv_list[0]
    concat3 = fluid.layers.concat([res2_unpool, res2], axis=1)
    concat3_conv1 = conv_bn_relu_layer(input=concat3, num_filters=64, filter_size=1, name='concat3_conv1')      # Todo 这里和论文中的不太一样
    concat3_conv3 = conv_bn_relu_layer(input=concat3_conv1, num_filters=32, filter_size=3, padding=1, name='concat3_conv3')

    last_feature = conv_bn_relu_layer(input=concat3_conv3, num_filters=32, filter_size=3, padding=1, name='last_feature')

    F_score = last_conv(input=last_feature, num_filters=1, filter_size=1)
    F_score = fluid.layers.sigmoid(F_score, name='F_score')

    geo_map = last_conv(input=last_feature, num_filters=4, filter_size=1)
    geo_map = fluid.layers.sigmoid(geo_map, name='geo_map') * 512   # 这里乘以512的作用是？

    angle_map = last_conv(input=last_feature, num_filters=1, filter_size=1)
    angle_map = fluid.layers.sigmoid(angle_map, name='angle_map')
    angle_map = (angle_map - 0.5) * math.pi / 2.

    F_geometry = fluid.layers.concat([geo_map, angle_map], axis=1)

    return F_score, F_geometry


def add_ResNet_roi_conv5_head(head_input, rois):    # head_input: [-1, 1024, 84, 84], roids: [-1, 4]
    if cfg.roi_func == 'RoIRool':
        pool = fluid.layers.roi_pool(input=head_input, rois=rois, pooled_height=cfg.roi_resolution, pooled_width=cfg.roi_resolution, spatial_scale=cfg.spatial_scale)
    elif cfg.roi_func == 'RoIAlign':
        pool = fluid.layers.roi_align(input=head_input, rois=rois, pooled_height=cfg.roi_resolution, pooled_width=cfg.roi_resolution, spatial_scale=cfg.spatial_scale, sampling_ratio=cfg.sampling_ratio)
    res5 = layer_warp(bottleneck, pool, 512, 3, 2, name="res5")      # pool: [-1, 1024, 14, 14]
    return res5     # res5: [-1, 2048, 7, 7]


def layer_warp(block_func, input, ch_out, count, stride, name):
    res_out = block_func(input, ch_out, stride, name=name + "a")
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1, name=name + chr(ord("a") + i))
    return res_out


def bottleneck(input, ch_out, stride, name):
    short = shortcut(input, ch_out * 4, stride, name=name + "_branch1")
    conv1 = conv_affine_layer(input, ch_out, 1, stride, 0, name=name + "_branch2a")
    conv2 = conv_affine_layer(conv1, ch_out, 3, 1, 1, name=name + "_branch2b")
    conv3 = conv_affine_layer(conv2, ch_out * 4, 1, 1, 0, act=None, name=name + "_branch2c")
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu', name=name + '.add.output.5')


def conv_affine_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      name=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(name=name + '_weights'),
        bias_attr=False,
        name=name + '.conv2d.output.1'
    )
    if name == "conv1":
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]

    scale = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(name=bn_name+'_scale', learning_rate=0.),
        default_initializer=Constant(1.)
    )
    scale.stop_gradient = True
    bias = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(bn_name+'_offset', learning_rate=0.),
        default_initializer=Constant(0.)
    )
    bias.stop_gradient = True

    out = fluid.layers.affine_channel(x=conv, scale=scale, bias=bias)
    if act == 'relu':
        out = fluid.layers.relu(x=out)
    return out


def shortcut(input, ch_out, stride, name):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        return conv_affine_layer(input, ch_out, 1, stride, 0, None, name=name)
    else:
        return input


def basicblock(input, ch_out, stride, name):
    short = shortcut(input, ch_out, stride, name=name)
    conv1 = conv_affine_layer(input, ch_out, 3, stride, 1, name=name)
    conv2 = conv_affine_layer(conv1, ch_out, 3, 1, 1, act=None, name=name)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name)


ResNet_cfg = {
    18: ([2, 2, 2, 1], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
}