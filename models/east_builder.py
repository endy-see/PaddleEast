import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from config import cfg
import pdb


class East(object):
    def __init__(self,
                 add_conv_body_func=None,
                 add_feature_merging_func=None,
                 mode='train',
                 use_pyreader=True,
                 use_random=True):
        self.add_conv_body_func = add_conv_body_func
        self.add_feature_merging_func = add_feature_merging_func
        self.mode = mode
        self.use_pyreader = use_pyreader
        self.use_random = use_random

    def build_model(self, image_shape):
        self.build_input(image_shape)
        body_conv = self.add_conv_body_func(self.image)     # body_conv: [-1, 1024, 84, 84]
        F_score, F_geometry = self.add_feature_merging_func(body_conv)
        self.pred_score = F_score
        self.pred_geometry = F_geometry
        if self.mode != 'train':
            self.eval_bbox()

    def loss(self):
        # classification loss
        loss_cls = self.classification_loss()
        # slanted rectangle loss
        loss_slanted_rectangle, loss_angle = self.slanted_loss()
        loss_geo = loss_slanted_rectangle + 20 * loss_angle
        loss = fluid.layers.sum(fluid.layers.mean(loss_geo * self.gt_score_map * self.gt_training_mask), loss_cls)

        rkeys = ['loss', 'loss_cls', 'loss_geo', 'loss_slanted_rectangle', 'loss_angle']
        rloss = [loss, loss_cls, loss_geo, loss_slanted_rectangle, loss_angle]
        return rloss, rkeys

    def classification_loss(self):
        # use dice coefficient
        eps = 1e-5
        intersection = fluid.layers.sum(self.gt_score_map * self.pred_score * self.gt_training_mask)
        union = fluid.layers.sum(self.gt_score_map * self.gt_training_mask) + fluid.layers.sum(self.pred_score * self.gt_training_mask) + eps
        loss = 1. - (2 * intersection / union)
        return loss * 0.01

    def slanted_loss(self):
        gt_dt, gt_dr, gt_db, gt_dl, self.gt_theta = fluid.layers.split(self.gt_geo_map, num_or_sections=1, dim=1)
        pred_dt, pred_dr, pred_db, pred_dl, self.pred_theta = fluid.layers.split(self.pred_geometry, num_or_sections=1, dim=1)
        gt_area = (gt_dt + gt_db) * (gt_dr + gt_dl)
        pred_area = (pred_dt + pred_db) * (pred_dr + pred_dl)

        w_union = fluid.layers.elementwise_min(gt_dr, pred_dr) + fluid.layers.elementwise_min(gt_dl, pred_dl)
        h_union = fluid.layers.elementwise_min(gt_dt, pred_dt) + fluid.layers.elementwise_min(gt_db, pred_db)

        area_intersect = w_union * h_union
        area_union = gt_area + pred_area - area_intersect

        L_AABB = -fluid.layers.log((area_intersect + 1.0) / (area_union + 1.0))
        theta_loss = 1 - fluid.layers.cos(self.pred_theta - self.gt_theta)

        return L_AABB, theta_loss

    def rpn_loss(self):
        rpn_cls_score_reshape = fluid.layers.transpose(self.rpn_cls_score, perm=[0, 2, 3, 1])
        rpn_bbox_pred_reshape = fluid.layers.transpose(self.rpn_bbox_pred, perm=[0, 2, 3, 1])
        anchor_reshape = fluid.layers.reshape(self.anchor, shape=(-1, 4))
        var_reshape = fluid.layers.reshape(self.var, shape=(-1, 4))

        rpn_cls_score_reshape = fluid.layers.reshape(x=rpn_cls_score_reshape, shape=(0, -1, 1))
        rpn_bbox_pred_reshape = fluid.layers.reshape(x=rpn_bbox_pred_reshape, shape=(0, -1, 4))
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = fluid.layers.rpn_target_assign(bbox_pred=rpn_bbox_pred_reshape, cls_logits=rpn_cls_score_reshape, anchor_box=anchor_reshape, anchor_var=var_reshape, gt_boxes=self.gt_box, is_crowd=self.is_crowd, im_info=self.im_info, rpn_batch_size_per_im=cfg.TRAIN.rpn_batch_size_per_im, rpn_straddle_thresh=cfg.TRAIN.rpn_straddle_thresh, rpn_fg_fraction=cfg.TRAIN.rpn_fg_fraction, rpn_positive_overlap=cfg.TRAIN.rpn_positive_overlap, rpn_negative_overlap=cfg.TRAIN.rpn_negative_overlap, use_random=self.use_random)
        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=score_pred, label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(rpn_cls_loss, name='loss_rpn_cls')

        rpn_reg_loss = fluid.layers.smooth_l1(x=loc_pred, y=loc_tgt, sigma=3.0, inside_weight=bbox_weight, outside_weight=bbox_weight)
        rpn_reg_loss = fluid.layers.reduce_sum(rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        rpn_reg_loss = rpn_reg_loss / norm

        return rpn_cls_loss, rpn_reg_loss

    def fast_rcnn_loss(self):
        labels_int64 = fluid.layers.cast(x=self.labels_int32, dtype='int64')
        labels_int64.stop_grdient = True
        loss_cls = fluid.layers.softmax_with_cross_entropy(logits=self.cls_score, label=labels_int64, numeric_stable_mode=True)
        loss_cls = fluid.layers.reduce_mean(loss_cls)

        loss_bbox = fluid.layers.smooth_l1(x=self.bbox_pred, y=self.bbox_targets, inside_weight=self.bbox_inside_weights, outside_weight=self.bbox_outside_weights, sigma=1.0)
        loss_bbox = fluid.layers.reduce_sum(loss_bbox)

        return loss_cls, loss_bbox

    def eval_bbox(self):
        self.im_scale = fluid.layers.slice(self.im_info, [1], starts=[2], ends=[3])
        im_scale_lod = fluid.layers.sequence_expand(self.im_scale, self.rpn_rois)
        boxes = self.rpn_rois / im_scale_lod
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
        bbox_pred_reshape = fluid.layers.reshape(self.bbox_pred, (-1, cfg.class_num, 4))
        decoded_box = fluid.layers.box_coder(prior_box=boxes, prior_box_var=cfg.bbox_reg_weights, target_box=bbox_pred_reshape, code_type='decode_center_size', box_normalized=False, axis=1)
        cliped_box = fluid.layers.box_clip(input=decoded_box, im_info=self.im_info)
        self.pred_result = fluid.layers.multiclass_nms(bboxes=cliped_box, scores=cls_prob, score_threshold=cfg.TEST.score_thresh, nms_top_k=-1, nms_threshold=cfg.TEST.nms_thresh, keep_top_k=cfg.TEST.detections_per_im, normalized=False)

    def build_input(self, image_shape):
        if self.use_pyreader:
            in_shapes = [[-1] + image_shape, [-1, 1], [-1, 4], [-1, 1]]
            lod_levels = [0, 1, 1, 1]
            dtypes = ['float32', 'float32', 'float32', 'int32']
            self.py_reader = fluid.layers.py_reader(capacity=64, shapes=in_shapes, lod_levels=lod_levels, dtypes=dtypes, use_double_buffer=True)
            ins = fluid.layers.read_file(self.py_reader)
            self.image = ins[0]
            self.gt_score_map = ins[1]
            self.gt_geo_map = ins[2]
            self.gt_training_mask = ins[3]      # mask究竟是int还是float
        else:
            self.image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
            self.gt_score_map = fluid.layers.data(name='gt_score_map', shape=[1], dtype='float32', lod_level=1)
            self.gt_geo_map = fluid.layers.data(name='gt_geo_map', shape=[4], dtype='float32', lod_level=1)
            self.gt_training_mask = fluid.layers.data(name='gt_training_mask', shape=[1], dtype='int32', lod_level=1)

    def feeds(self):
        if self.mode == 'infer':
            return [self.image]
        if self.mode == 'val':
            return [self.image]
        return [self.image, self.gt_score_map, self.gt_geo_map, self.gt_training_mask]
