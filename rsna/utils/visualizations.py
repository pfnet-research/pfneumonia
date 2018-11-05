import numpy as np

import chainercv
from chainercv.visualizations import vis_bbox


def vis_pred_gt_bboxes(image, pred_bbox, gt_bbox, ax=None, show_iou=True):
    """Visualize both prediction and GT bboxes.

    The colors of prediction and GT bboxes are red and green, respectively.

    Args:
        image (~np.ndarray): Greyscale image of shape (H, W).
        pred_bbox (~np.ndarray): Prediction bbox of shape (R, 4).
        gt_bbox (~np.ndarray): GT bbox of shape (R', 4).
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
        show_iou (bool): If `True`, show maximum IoU for each prediction bbox.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    image = np.repeat([image], 3, axis=0)  # (3, H, W)

    if show_iou and len(pred_bbox) > 0 and len(gt_bbox) > 0:
        iou = chainercv.utils.bbox_iou(pred_bbox, gt_bbox)
        iou = iou.max(axis=1)

        label_names = ['%.3f' % x for x in iou]
        label = range(len(label_names))
    else:
        label_names, label = None, None

    ax = vis_bbox(image, pred_bbox, label=label, linewidth=1, ax=ax, label_names=label_names)
    transp_image = np.zeros((4, *image.shape[1:]), dtype=np.float32)
    ax = vis_bbox(transp_image, gt_bbox, instance_colors=((0, 255, 0),), ax=ax, linewidth=1)
    return ax
