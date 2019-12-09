from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from config import cfg
import os
import glob as gb
import numpy as np
import cv2
import sys
import csv
import math
from shapely.geometry import Polygon


class DatasetPath(object):
    def __init__(self, mode):
        self.mode = mode

    def get_images_and_annos(self):
        """Get image's path and name"""
        img_dir = os.path.join(cfg.data_dir, self.mode, 'img')
        anno_dir = os.path.join(cfg.data_dir, self.mode, 'gt')
        anno_list = os.listdir(anno_dir)

        img_path_list = []
        for ext in ['jpg', 'png', 'jpeg', 'JPG']:
            src_img_list = gb.glob(os.path.join(img_dir, '*.{}'.format(ext)))
            img_path_list.extend([img_path for img_path in src_img_list if os.path.basename(img_path)[:-4]+'.txt' in anno_list])
        img_name_list = []
        for i in range(len(img_path_list)):
            img_name_list.append(img_path_list[i].split('/')[-1])

        # check
        for i in range(len(img_path_list)):
            assert os.path.basename(img_path_list[i]) == img_name_list[i], 'image path cannot corresponding to img name'
        print('EAST <==> Prepare <==> Total:{} images for train'.format(len(img_path_list)))

        img_path_list = sorted(img_path_list)
        img_name_list = sorted(img_name_list)

        txt_path_list = []
        txt_path_list.extend([os.path.join(anno_dir, txt_name[:-4]+'.txt') for txt_name in img_name_list])
        txt_name_list = []
        for i in range(len(txt_path_list)):
            txt_name_list.append(txt_name_list[i].split('/')[-1])
        return img_path_list, img_name_list, txt_path_list, txt_name_list


class IcdarDataset(object):
    """A class representing a icdar txt dataset"""
    def __init__(self, mode):
        print('Creating: {}'.format(cfg.dataset))
        self.name = cfg.dataset
        self.is_train = mode == 'train'
        data_path = DatasetPath.mode(mode)
        self.img_path_list, self.img_name_list, self.txt_path_list, self.txt_name_list = data_path.get_images_and_annos()

    def data_nums(self):
        return len(self.img_name_list)

    def getitem(self, index):
        img, score_map, geo_map, training_mask = image_label(self.img_path_list, self.txt_path_list, index)
        img = img.transpose(2, 0, 1)
        return (img, score_map, geo_map, training_mask)


def image_label(image_path_list, txt_path_list, index, input_size=512, random_scale=np.array([0.5, 1, 2.0, 3.0]), background_ratio=3. / 8.):
    """
    Get image's corresponding matrix and ground truth
    images  [512, 512, 3]
    score   [128, 128, 1]
    geo     [128, 128, 5]
    mask    [128, 128, 1]
    """
    try:
        im_fn = image_path_list[index]
        txt_fn = txt_path_list[index]

        im = cv2.imread(im_fn)
        h, w, _ = im.shape
        if not os.path.exists(txt_fn):
            sys.exit('text file {} does not exists.'.format(txt_fn))
        text_polys, text_tags = load_annotation(txt_fn)
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

        # random scale this image
        rd_scale = np.random.choice(random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        h, w, _ = im.shape

        # pad the image to the training input size or the longer side of image
        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = im_padded

        # resize the image to input size
        new_h, new_w, _ = im.shape
        resize_h = input_size
        resize_w = input_size
        im = cv2.resize(im, dsize=(resize_w, resize_h))
        resize_ratio_3_x = resize_w / float(new_w)
        resize_ratio_3_y = resize_h / float(new_h)
        # text_polys[:, :, 0] *= resize_ratio_3_x
        # text_polys[:, :, 1] *= resize_ratio_3_y
        for i in range(len(text_polys)):
            for j in range(4):
                text_polys[i][j][0] *= resize_ratio_3_x
                text_polys[i][j][1] *= resize_ratio_3_y

        if np.random.rand() < background_ratio:
            # crop background
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
            assert len(text_polys) == 0, 'crop area should have no text polys'
            # re pad and resize the croped image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(input_size, input_size))
            scpre_map = np.zeros((input_size, input_size), dtype=np.uint8)
            geo_map = np.zeros((input_size, input_size, 5), dtype=np.float32)
            training_mask = np.ones((input_size, input_size), dtype=np.uint8)
        else:
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
            if len(text_polys) == 0:
                # for som reason, gt contains no polys, have to return black
                score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                geo_map = np.zeros((input_size, input_size, 5), dtype=np.float32)
                training_mask = np.zeros((input_size, input_size), dtype=np.uint8)
                images = im[:, :, ::-1].astype(np.float32)
                score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
                geo_maps = geo_map[::4, ::4, :].astype(np.float32)
                training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)
                return images, score_maps, geo_maps, training_masks
            # pad the croped image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = input_size
            resize_w = input_size
            im = cv2.resize(im, dsize=(resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            for i in range(len(text_polys)):
                for j in range(4):
                    text_polys[i][j][0] *= resize_ratio_3_x
                    text_polys[i][j][1] *= resize_ratio_3_y
            # text_polys[:, :, 0] *= resize_ratio_3_x
            # text_polys[:, :, 1] *= resize_ratio_3_y
            new_h, new_w, _ = im.shape
            score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
    except Exception as e:
        print('Exception continue')
        return None, None, None, None

    images = im[:, :, ::-1].astype(np.float32)
    score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)     # 图像大小原本是512，这里每隔4个像素取一个，所以最后maps的大小就是原来的1/4
    geo_maps = geo_map[::4, ::4, :].astype(np.float32)
    training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

    return images, score_maps, geo_maps, training_masks


def load_annotation(txt_path):
    """
    Load annotation from the text file.
    Note:
        1. top left vertice
        2. clockwise
    """
    text_polys = []
    text_tags = []
    if not os.path.exists(txt_path):
        return np.array(text_polys, dtype=np.float32)
    with open(txt_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]    # strip BOM. \ufeff for python3, \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)

    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def check_and_validate_polys(polys, tags, hw):
    """Check so that the text poly is in the same direction, and also filter some invalid polygons"""
    if polys.shape[0] == 0:
        return polys

    (h, w) = hw
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []

    # find top-left and clockwise
    polys = choose_best_begin_point(polys)

    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            continue
        if p_area > 0:
            # poly in wrong direction
            poly = poly[(0, 3, 2, 1), :]    # 用polygon_area求出的面积正常的话应该小于0
        validated_polys.append(poly)
        validated_tags.append(tag)
    return validated_polys, validated_tags


def polygon_area(poly):
    """Compute area of a polygon"""
    poly_ = np.array(poly)      # 其实这里进来的poly已经是np了，不过再做一次np.array也不影响结果
    assert poly_.shape == (4, 2), 'poly shape should be 4,2'
    edge = [(poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])]
    return np.sum(edge) / 2.


def choose_best_begin_point(polys):
    """find top-left vertice and resort"""
    final_result = []
    for coordinate in polys:
        x1 = coordinate[0][0]
        y1 = coordinate[0][1]
        x2 = coordinate[1][0]
        y2 = coordinate[1][1]
        x3 = coordinate[2][0]
        y3 = coordinate[2][1]
        x4 = coordinate[3][0]
        y4 = coordinate[3][1]
        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)
        combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                     [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                     [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                     [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
        dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        total_distance = 100000000.0
        flag = 0
        for i in range(4):
            temp_distance_sum = calculate_distance(combinate[i][0], dst_coordinate[0]) + \
                         calculate_distance(combinate[i][1], dst_coordinate[1]) + \
                         calculate_distance(combinate[i][2], dst_coordinate[2]) + \
                         calculate_distance(combinate[i][3], dst_coordinate[3])
            if temp_distance_sum < total_distance:
                total_distance = temp_distance_sum
                flag = i
        final_result.append(combinate[flag])
    return final_result


def calculate_distance(point1, point2):
    return math.sqrt(math.pow(point1[0]-point2[0], 2), math.pow(point1[1]-point2[1], 2))


def crop_area(im, polys, tags, crop_background=False, max_tries=5000):
    """make random crop from the input image"""
    if polys.shape[0] == 0:
        return im, [], []

    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w: maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h: maxy + pad_h] = 1

    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < 0.1 * w or ymax - ymin < 0.1 * h:
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                im = im[ymin:ymax+1, xmin:xmax+1, :]
                polys = []
                tags = []
                return im, polys, tags
            else:
                continue
        else:
            if not crop_background:
                im = im[ymin:ymax+1, xmin:xmax+1, :]
                polys = polys.tolist()
                polys = [polys[i] for i in selected_polys]
                polys = np.array(polys)
                polys[:, :, 0] -= xmin
                polys[:, :, 1] -= ymin
                polys = polys.astype(np.int32)
                polys = polys.tolist()

                tags = tags.tolist()
                tags = [tags[i] for i in selected_polys]
                return im, polys, tags
            else:
                continue
    # if try max times still not get a result, there will return the original input info
    return im, polys, tags


def generate_rbox(im_size, polys, tags):
    """
    Generate rbox for training
    score map is (128, 128, 1)
    poly mask is (128, 128, 1) with different colors
    geo map is (128, 128, 5) with distance to top,right,bottom,left
    """
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during training, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]
        poly = np.array(poly)
        tag = np.array(tag)
        r = [None, None, None, None]    # 每个点对应的最短边边长
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        # use different color to draw poly mask
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < 10:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        # generate parallelograms
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]

            # fit_line([x1, x2], [y1, y2]) return k, -1, b just a line
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])             # p0, p1
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])    # p0, p3
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])     # p1, p2

            # select shorter line
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # parallel line pass p2
                if edge[1] == 0:    # verticle
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # parallel line pass p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite[edge[0], -1, p3[1] - edge[0] * p3[0]]

            # move forward edge
            new_p1 = p1
            new_p2 = line_cross_point(edge_opposite, forward_edge)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:    # vertical
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])

            # or move backward edge
            new_p0 = p0
            new_p3 = line_cross_point(edge_opposite, backward_edge)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])

        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort this polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)     # sum every point coordinate
        min_coord_idx = np.argmin(parallelogram_coord_sum)      # select the minimum sum point as lt
        parallelogram = parallelogram[[min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectangle = rectangle_from_parallelogram(parallelogram)
        rectangle, rotate_angle = sort_rectangle(rectangle)
        p0_rect, p1_rect, p2_rect, p3_rect = rectangle
        # this is one area of many
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # bottom
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴，那么p0为左上角
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点
        p_lowest_right = (p_lowest - 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index - 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def rectangle_from_parallelogram(parallelogram):
    """Fit a rectangele from a parallelogram"""
    p0, p1, p2, p3 = parallelogram
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)
            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)
            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            # p1 and p3
            ## p1
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[2]])
            p1p2_verticle = line_verticle(p1p2, p0)
            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            ## p3
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)
            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)
            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)
            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            # p0 and p2
            # p0
            p0p3 = fit_line([p0[0], p3[0]], [p1[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)
            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            # p2
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)
            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1, point[1] - (-1.0 / line[0] * point[0])]
    return verticle


def line_cross_point(line1, line2):
    # line1 0=ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        # the slopes of the two lines are the same
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]   # vertical
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    distance = 0
    try:
        eps = 1e-5
        distance = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / (np.linalg.norm(p2 - p1) + eps)
    except:
        print('point dist to line raise Exception')
    return distance


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:              # 垂直
        return [1., 0., -p1[0]]     # 这是什么原理。。
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def shrink_poly(poly, r):
    """
    Fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    """
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][0] -= R * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        # first deal longer side: p0p3 & p1p2, then p0p1 & p3p2
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        # p3, p2
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0], poly[3][0]))
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
    return poly


