import numpy as np
from PIL import Image, ImageDraw


def classifications_to_rails(clf, classes):
    """Processes the classification model's output to an array of left and right rail point relative coordinates (x, y).

    Args:
        clf (numpy.ndarray): Argmax class indices for each rail at each anchor. Shape of (2, H).
        classes (int): Number of classes (grid cells) for each anchor (background class not included).

    Returns:
        numpy.ndarray: Ego-path as an array of left and right rail points. Shape of (2, N, 2) with N <= H.
    """
    limit_idx = np.where(clf == classes)[1]
    limit_idx = limit_idx.min() if limit_idx.size > 0 else clf.shape[1]
    switched_idx = np.where(clf[0, :] >= clf[1, :])[0]
    switched_idx = switched_idx[0] if switched_idx.size > 0 else clf.shape[1]
    crop_idx = min(limit_idx, switched_idx)
    xrails = clf[:, :crop_idx] / (classes - 1)
    yrails = np.linspace(1, 0, clf.shape[1])[:crop_idx]
    rails = [np.column_stack((xrails[i, :], yrails)) for i in range(xrails.shape[0])]
    return np.array(rails)


def regression_to_rails(traj, ylim):
    """Processes the regression model's output to an array of left and right rail point relative coordinates (x, y).

    Args:
        traj (numpy.ndarray): Regressed x-coordinates for each rail at each anchor. Shape of (2, H).
        ylim (float): Regressed y-limit of the ego-path (between 0 and 1)

    Returns:
        numpy.ndarray: Ego-path as an array of left and right rail points. Shape of (2, N, 2) with N <= H.
    """
    limit_idx = round(ylim * traj.shape[1])  # convert ylim to index
    switched_idx = np.where(traj[0, :] >= traj[1, :])[0]  # if left rail >= right rail
    switched_idx = switched_idx[0] if switched_idx.size > 0 else traj.shape[1]
    crop_idx = min(limit_idx, switched_idx)
    xrails = np.clip(traj[:, :crop_idx], 0, 1)  # clip traj points to image bounds
    yrails = np.linspace(1, 0, traj.shape[1])[:crop_idx]
    rails = [np.column_stack((xrails[i, :], yrails)) for i in range(xrails.shape[0])]
    return np.array(rails)

# FIXME: Jn 修改 适配 ylim [start,stop] 二维输出
# def regression_to_rails(traj, ylim):
#     """Processes the regression model's output to an array of left and right rail point relative coordinates (x, y).

#     Args:
#         traj (numpy.ndarray): Regressed x-coordinates for each rail at each anchor. Shape of (2, H).
#         ylim (float | array-like): (y_start, y_end) in [0, 1] (0=top, 1=bottom), or a single float limit.

#     Returns:
#         numpy.ndarray: Ego-path as an array of left and right rail points. Shape of (2, N, 2) with N <= H.
#     """
#     H = traj.shape[1]
#     ylim_arr = np.asarray(ylim, dtype=np.float32)

#     # 锚点索引 i=0 对应底部 (y=1.0)，i=H-1 对应顶部 (y=0.0)
#     # yrails 是从 1.0 到 0.0 的数组，长度 H
#     if ylim_arr.ndim == 0:
#         # 兼容单值情况：视为从底部到某个上限比例
#         y_start, y_end = 0.0, float(np.clip(ylim_arr.item(), 0.0, 1.0))
#         start_idx = 0
#         end_idx = int(np.clip(round(y_end * H), 0, H))
#     else:
#         # [y_start, y_end] 为从顶部向下的比例（0=顶，1=底）
#         y_start = float(np.clip(ylim_arr[0], 0.0, 1.0))
#         y_end = float(np.clip(ylim_arr[1], y_start, 1.0))
#         # 映射到 i=0(底, 1.0) -> i=H-1(顶, 0.0) 的索引区间
#         idx_from_top_1 = int(np.clip(round((1.0 - y_end) * (H - 1)), 0, H - 1))
#         idx_from_top_2 = int(np.clip(round((1.0 - y_start) * (H - 1)), 0, H - 1))
#         start_idx = min(idx_from_top_1, idx_from_top_2)
#         end_idx = min(max(idx_from_top_1, idx_from_top_2) + 1, H)

#     switched_idx = np.where(traj[0, :] >= traj[1, :])[0]  # if left rail >= right rail
#     switched_idx = int(switched_idx[0]) if switched_idx.size > 0 else H

#     start_idx = min(start_idx, switched_idx)
#     end_idx = min(end_idx, switched_idx)

#     if end_idx <= start_idx:
#         return np.zeros((2, 0, 2), dtype=np.float32)

#     xrails = np.clip(traj[:, start_idx:end_idx], 0.0, 1.0)
#     yrails = np.linspace(1.0, 0.0, H)[start_idx:end_idx]
#     rails = [np.column_stack((xrails[i, :], yrails)) for i in range(xrails.shape[0])]
#     return np.array(rails, dtype=np.float32)


def scale_rails(rails, crop_coords, img_shape):
    """Scales the rail point relative coordinates to the target image shape with consideration of the cropping coordinates.

    Args:
        rails (numpy.ndarray): Array of left and right rail points.
        crop_coords (tuple or None): Inclusive absolute coordinates (xmin, ymin, xmax, ymax) of the cropped region.
        img_shape (tuple): Shape (W, H) of the target image.

    Returns:
        numpy.ndarray: Ego-path points absolute coordinates in the target image.
    """
    if crop_coords is not None:
        width = crop_coords[2] - crop_coords[0]
        height = crop_coords[3] - crop_coords[1]
        rails[:, :, 0] = rails[:, :, 0] * width + crop_coords[0]
        rails[:, :, 1] = rails[:, :, 1] * height + crop_coords[1]
    else:
        rails[:, :, 0] *= img_shape[0] - 1
        rails[:, :, 1] *= img_shape[1] - 1
    return rails


def rails_to_mask(rails, mask_shape):
    """Creates a binary mask of the detected region from the ego-path points.

    Args:
        rails (list): List containing the left and right rails lists of rails point coordinates (x, y).
        mask_shape (tuple): Shape (W, H) of the target mask.

    Returns:
        PIL.Image.Image: Binary mask of the detected region.

    """
    left_rail, right_rail = rails
    if not left_rail or not right_rail:
        return np.zeros(mask_shape[::-1], dtype=np.uint8)
    mask = Image.new("L", mask_shape, 0)
    draw = ImageDraw.Draw(mask)
    points = left_rail + right_rail[::-1]
    draw.polygon([tuple(xy) for xy in points], fill=255)
    return mask


def scale_mask(mask, crop_coords, img_shape):
    """Scales the mask to the target image shape with consideration of the cropping coordinates.

    Args:
        mask (PIL.Image.Image): Binary mask to be scaled.
        crop_coords (tuple or None): Inclusive absolute coordinates (xmin, ymin, xmax, ymax) of the cropped region.
        img_shape (tuple): Shape (W, H) of the target image.

    Returns:
        PIL.Image.Image: Scaled mask.
    """
    if crop_coords is not None:
        xleft, ytop, xright, ybottom = crop_coords
        mask = mask.resize((xright - xleft + 1, ybottom - ytop + 1), Image.NEAREST)
        rescaled_mask = Image.new("L", img_shape, 0)
        rescaled_mask.paste(mask, (xleft, ytop))
    else:
        rescaled_mask = mask.resize(img_shape, Image.NEAREST)
    return rescaled_mask
