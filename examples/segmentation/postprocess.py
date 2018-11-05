import matplotlib.pyplot as plt
import numpy as np
import os

import chainer.functions as F
from chainercv.visualizations import vis_image

from rsna.utils.visualizations import vis_pred_gt_bboxes


def get_center_x(image):
    image = image[0]
    v_sums = np.sum(image, axis=0)
    whole_sum = np.sum(v_sums)
    return np.sum(np.arange(len(v_sums)) * v_sums) / whole_sum


class Postprocessor(object):

    def __init__(self, downscale, score_thresh, size_thresh, edge_conf_operation):
        self.downscale = downscale
        self.score_thresh = score_thresh
        self.size_thresh = size_thresh
        assert edge_conf_operation in ('max', 'mean', 'logmean')
        self.edge_conf_operation = edge_conf_operation

    def __call__(self, image, h_seg, h_hor, h_ver):
        """

        Args:
            image (~np.ndarray): (1, 1024, 1024)
            h_seg (~np.ndarray): (1, H, W)
            h_hor (~np.ndarray): (2, H, W)
            h_ver (~np.ndarray): (2, H, W)

        Returns:

        """
        center_x = int(get_center_x(image) / (self.downscale * 2) + 0.5)
        bbox_right_lung, score_right_lung = self.compute_bbox_and_score(h_seg[:, :, :center_x],
                                                                        h_hor[:, :, :center_x],
                                                                        h_ver[:, :, :center_x])
        bbox_left_lung, score_left_lung = self.compute_bbox_and_score(h_seg[:, :, center_x:],
                                                                      h_hor[:, :, center_x:],
                                                                      h_ver[:, :, center_x:])
        bbox_left_lung[:, 1::2] += center_x
        bbox = np.vstack((bbox_right_lung, bbox_left_lung)).astype(np.float32)
        bbox = (bbox + 0.5) * self.downscale * 2
        label = np.zeros((len(bbox), ), dtype=np.float32)
        score = np.array([x for x in (score_right_lung, score_left_lung) if x > 0],
                         dtype=np.float32)
        return bbox, label, score

    def postprocess(self, in_data):
        patient_id, image, h_seg, h_hor, h_ver = in_data
        print('postprocessing {}...'.format(patient_id))
        bbox, label, score = self.__call__(image, h_seg, h_hor, h_ver)
        return patient_id, bbox, label, score

    def compute_bbox_and_score(self, h_seg, h_hor, h_ver):
        """

        Args:
            h_seg (~np.ndarray): (1, H, W)
            h_hor (~np.ndarray): (2, H, W)
            h_ver (~np.ndarray): (2, H, W)

        Returns:

        """
        hor_edge_conf = F.sigmoid(h_hor).array  # (2, H, W)
        ver_edge_conf = F.sigmoid(h_ver).array  # (2, H, W)

        bbox, prob = self.find_best_box(hor_edge_conf, ver_edge_conf)
        if bbox is None or prob < self.score_thresh:
            return np.empty((0, 4), dtype=np.float32), 0

        # `prob` is considered as the score
        return np.array(bbox, dtype=np.float32).reshape(1, 4), prob

    def find_best_box(self, hor_edge_conf, ver_edge_conf):
        """

        Args:
            hor_edge_conf (~np.ndarray): (2, H, W)
            ver_edge_conf (~np.ndarray): (2, H, W)

        Returns:

        """
        hor_max = hor_edge_conf.max(axis=2)  # (2, H)
        ver_max = ver_edge_conf.max(axis=1)  # (2, W)
        t_max = hor_max[0].max()
        b_max = hor_max[1].max()
        l_max = ver_max[0].max()
        r_max = ver_max[1].max()
        upper_prob = t_max * b_max * l_max * r_max
        if upper_prob < self.score_thresh:
            return None, 0

        t_count = (hor_max[0] >= self.score_thresh / (upper_prob / t_max)).sum()
        b_count = (hor_max[1] >= self.score_thresh / (upper_prob / b_max)).sum()
        l_count = (ver_max[0] >= self.score_thresh / (upper_prob / l_max)).sum()
        r_count = (ver_max[1] >= self.score_thresh / (upper_prob / r_max)).sum()
        comb = t_count * b_count * l_count * r_count
        if comb == 0:
            return None, 0

        T = hor_max[0].argsort()[:-1 - t_count:-1]
        B = hor_max[1].argsort()[:-1 - b_count:-1]
        L = ver_max[0].argsort()[:-1 - l_count:-1]
        R = ver_max[1].argsort()[:-1 - r_count:-1]

        _, H, W = hor_edge_conf.shape
        hor_cumsum = np.zeros((2, H + 1, W + 1), dtype=np.float64)
        ver_cumsum = np.zeros((2, H + 1, W + 1), dtype=np.float64)
        if self.edge_conf_operation == 'logmean':
            hor_cumsum[:, 1:, 1:] = np.log(np.clip(hor_edge_conf, 1e-8, None))
            ver_cumsum[:, 1:, 1:] = np.log(np.clip(ver_edge_conf, 1e-8, None))
        else:
            hor_cumsum[:, 1:, 1:] = hor_edge_conf
            ver_cumsum[:, 1:, 1:] = ver_edge_conf
        hor_cumsum = hor_cumsum.cumsum(axis=2)
        ver_cumsum = ver_cumsum.cumsum(axis=1)

        best_prob = self.score_thresh
        best_bbox = None
        for y_min in T:
            if hor_max[0, y_min] * b_max * l_max * r_max < best_prob:
                break
            for y_max in B:
                if hor_max[0, y_min] * hor_max[1, y_max] * l_max * r_max < best_prob:
                    break
                if y_max - y_min < self.size_thresh // (self.downscale * 2):
                    continue
                for x_min in L:
                    if hor_max[0, y_min] * hor_max[1, y_max] * ver_max[0, x_min] * r_max < best_prob:
                        break
                    for x_max in R:
                        if hor_max[0, y_min] * hor_max[1, y_max] * ver_max[0, x_min] * ver_max[1, x_max] < best_prob:
                            break
                        if x_max - x_min < self.size_thresh // (self.downscale * 2):
                            continue

                        if self.edge_conf_operation == 'max':
                            t_prob = hor_edge_conf[0, y_min, x_min:x_max + 1].max()
                            b_prob = hor_edge_conf[1, y_max, x_min:x_max + 1].max()
                            l_prob = ver_edge_conf[0, y_min:y_max + 1, x_min].max()
                            r_prob = ver_edge_conf[1, y_max:y_max + 1, x_max].max()
                        else:
                            t_prob = (hor_cumsum[0, y_min + 1, x_max + 1] - hor_cumsum[0, y_min + 1, x_min]) / (x_max - x_min + 1)
                            b_prob = (hor_cumsum[1, y_max + 1, x_max + 1] - hor_cumsum[1, y_max + 1, x_min]) / (x_max - x_min + 1)
                            l_prob = (ver_cumsum[0, y_max + 1, x_min + 1] - ver_cumsum[0, y_min, x_min + 1]) / (y_max - y_min + 1)
                            r_prob = (ver_cumsum[1, y_max + 1, x_max + 1] - ver_cumsum[1, y_min, x_max + 1]) / (y_max - y_min + 1)
                            if self.edge_conf_operation == 'logmean':
                                t_prob = np.exp(t_prob)
                                b_prob = np.exp(b_prob)
                                l_prob = np.exp(l_prob)
                                r_prob = np.exp(r_prob)
                        prob = t_prob * b_prob * l_prob * r_prob
                        if prob > best_prob:
                            best_prob = prob
                            best_bbox = (y_min, x_min, y_max, x_max)

        return best_bbox, best_prob


class DemoSaver(object):

    def __init__(self, dst_dir, downscale, overlay_seg):
        self.dst_dir = dst_dir
        self.downscale = downscale
        self.overlay_seg = overlay_seg

    def save(self, in_data):
        result, output, gt_bbox = in_data
        patient_id, bbox, label, score = result
        _, image, h_seg, h_hor, h_ver = output
        _, gt_bbox = gt_bbox

        ax = vis_pred_gt_bboxes(image[0], bbox, gt_bbox)

        if self.overlay_seg:
            seg = F.sigmoid(h_seg[0, :, :]).array
            seg = np.repeat(seg, self.downscale * 2, axis=0)
            seg = np.repeat(seg, self.downscale * 2, axis=1)
            mask_image = np.stack((np.zeros_like(seg),
                                   np.zeros_like(seg),
                                   np.zeros_like(seg) + 255,
                                   seg * 255)).astype(np.uint8)
            vis_image(mask_image, ax=ax)

        file_path = os.path.join(self.dst_dir, '{}.png'.format(patient_id))
        print("Saving demo to '{}'...".format(file_path))
        plt.savefig(file_path)
        plt.close()
