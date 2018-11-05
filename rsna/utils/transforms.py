def rotate_bbox(bbox, k, size):
    """Apply `rot90`s to bbox.

    Args:
        bbox (~np.ndarray): Bbox of shape (R, 4).
        k (int): # of rot90s counter-clockwise.
        size (tuple of int): Image's size (H, W) *BEFORE* rotation.
    """
    k %= 4
    H, W = size

    if k == 0:
        pass
    elif k == 1:
        bbox = bbox[:, [3, 0, 1, 2]]
        bbox[:, 0::2] = W - bbox[:, 0::2]
    elif k == 2:
        raise ValueError('Currently 180 degrees rotation is not supported.')
    else:
        bbox = bbox[:, [1, 2, 3, 0]]
        bbox[:, 1::2] = H - bbox[:, 1::2]

    return bbox
