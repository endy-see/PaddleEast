from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from config import cfg
from data_utils import DatasetPath
from pycocotools.coco import COCO
import copy
import os
import numpy as np
import box_utils


class JsonDataset(object):
    """A class representing a COCO json dataset"""
    def __init__(self, mode):
        print('Creating: {}'.format(cfg.dataset))
        self.name = cfg.dataset
        self.is_train = mode == 'train'
        data_path = DatasetPath(mode)
        data_dir = data_path.get_data_dir()
        file_list = data_path.get_file_list()
        self.image_directory = data_dir
        self.COCO = COCO(file_list)
        # set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map =dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {v: i+1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}

    def get_roidb(self):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)
        if self.is_train:
            # include ground-truth object annotations
            for entry in roidb:
                self._add_gt_annotations(entry)
            if cfg.TRAIN.use_flipped:
                self._extend_with_flipped_entries(roidb)
            roidb = self._filter_for_training(roidb)
        return roidb

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry"""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)

        valid_objs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj['area'] < cfg.TRAIN.gt_min_area:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue

            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(x1, y1, x2, y2, height, width)

            # require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        num_valid_objs = len(valid_objs)
        gt_boxes = np.zeros((num_valid_objs, 4), dtype=entry['gt_boxes'].dtype)
        gt_id = np.zeros((num_valid_objs), dtype=np.int64)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            gt_boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            gt_id[ix] = np.int64(obj['id'])
            is_crowd[ix] = obj['iscrowd']
        entry['gt_boxes'] = np.append(entry['gt_boxes'], gt_boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['gt_id'] = np.append(entry['gt_id'], gt_id)
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)

    def _prep_roidb_entry(self, entry):
        """Prepare roiddb entry"""
        im_path = os.path.join(self.image_directory, entry['file_name'])
        entry['image'] = im_path
        entry['flipped'] = False
        # empty placeholders
        entry['gt_boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_id'] = np.empty((0), dtype=np.int32)
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # remove unwanted fields that come from the json file (if they exist)
        for k in ['data_captured', 'url', 'license', 'file_name', 'license', 'coco_url', 'flickr_url']:
            if k in entry:
                del entry[k]

    def _extend_with_flipped_entries(self, roidb):
        """Flip each entry in the given roidb and return a new roidb that is the concatenation of the original roidb and the flipped entries"""
        flipped_roidb = []
        for entry in roidb:
            width = entry['width']
            gt_boxes = entry['gt_boxes'].copy()
            oldx1 = gt_boxes[:, 0].copy()
            oldx2 = gt_boxes[:, 2].copy()
            gt_boxes[:, 0] = width - oldx2 - 1
            gt_boxes[:, 2] = width - oldx1 - 1
            assert (gt_boxes[:, 2] >= gt_boxes[:, 0]).all()

            flipped_entry = {}
            dont_copy = ('gt_boxes', 'flipped', 'segms')
            for k, v in entry.items():
                if k not in dont_copy:
                    flipped_entry[k] = v

            flipped_entry['gt_boxes'] = gt_boxes
            flipped_entry['flipped'] = True

            flipped_roidb.append(flipped_entry)
        roidb.extend(flipped_roidb)

    def _filter_for_training(self, roidb):
        """Remove roidb entries that have no usable RoIs based on config settings."""
        def is_valid(entry):
            gt_boxes = entry['gt_boxes']
            valid = len(gt_boxes) > 0
            return valid

        num = len(roidb)
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        return filtered_roidb
