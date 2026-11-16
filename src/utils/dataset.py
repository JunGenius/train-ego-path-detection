import json
import os

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from .common import to_scaled_tensor
from .postprocessing import regression_to_rails


class PathsDataset(Dataset):
    def __init__(
        self,
        imgs_path,
        annotations_path,
        indices,
        config,
        method,
        img_aug=False,
        to_tensor=False,
    ):
        """Initializes the dataset for ego-path detection.

        Args:
            imgs_path (str): Path to the images directory.
            annotations_path (str):  Path to the annotations file.
            indices (list): List of indices to use in the dataset.
            config (dict): Data generation configuration.
            method (str): Method to use for ground truth generation ("classification", "regression" or "segmentation").
            img_aug (bool, optional): Whether to use stochastic image adjustment (brightness, contrast, saturation and hue). Defaults to False.
            to_tensor (bool, optional): Whether to return a ready to infer tensor (scaled and possibly resized). Defaults to False.
        """
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.imgs_path = imgs_path
        with open(annotations_path) as json_file:
            self.annotations = json.load(json_file)
        self.imgs = [sorted(self.annotations.keys())[i] for i in indices]
        self.config = config
        self.method = method

        self.img_aug = (
            transforms.ColorJitter(
                self.config["brightness"],
                self.config["contrast"],
                self.config["saturation"],
                self.config["hue"],
            )
            if img_aug
            else None
        )

        self.to_tensor = (
            transforms.Compose(
                [
                    to_scaled_tensor,
                    transforms.Resize(self.config["input_shape"][1:][::-1]),
                ]
            )
            if to_tensor
            else None
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.imgs_path, img_name))
        annotation = self.annotations[img_name]
        rails_mask = self.generate_rails_mask(img.size, annotation)
        img, rails_mask = self.random_crop(img, rails_mask)
        img, rails_mask = self.random_flip_lr(img, rails_mask)
        if self.to_tensor:
            img = self.to_tensor(img)
        if self.img_aug:
            img = self.img_aug(img)
        if self.method == "regression":
            path_gt, ylim_gt = self.generate_target_regression(rails_mask)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
                ylim_gt = torch.tensor(ylim_gt)
            return img, path_gt, ylim_gt
        elif self.method == "classification":
            path_gt = self.generate_target_classification(rails_mask)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
            return img, path_gt
        elif self.method == "segmentation":
            segmentation = self.generate_target_segmentation(rails_mask)
            if self.to_tensor:
                segmentation = segmentation.resize(
                    self.config["input_shape"][1:][::-1], Image.NEAREST
                )
                segmentation = to_scaled_tensor(segmentation)
            return img, segmentation
    # FIXME: Jn
    def _save_debug_vis(self, img, path_gt, ylim_gt, img_name, save_dir="debug_vis"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 将 img 还原为 uint8 的 numpy 数组 (H, W, C)
        if isinstance(img, torch.Tensor):
            # 如果做了 Normalize，需先反归一化（此处假设范围在 [0, 1]）
            vis_img = img.permute(1, 2, 0).cpu().numpy()
            vis_img = np.clip(vis_img * 255.0, 0, 255).astype(np.uint8)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        else:
            vis_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        h, w, _ = vis_img.shape
        
        # 2. 绘制 ylim_gt 红线
        ylim_val = ylim_gt.item() if isinstance(ylim_gt, torch.Tensor) else float(ylim_gt)
        ylim_px = int((1.0 - ylim_val) * h)
        # 限制在图像范围内防止越界
        ylim_px = max(0, min(ylim_px, h - 1))
        cv2.line(vis_img, (0, ylim_px), (w, ylim_px), (0, 0, 255), 2)

        # 3. 精确还原 path_gt (traj_target) 的锚点行坐标
        if isinstance(path_gt, torch.Tensor):
            path_gt = path_gt.cpu().numpy()
            
        anchors = path_gt.shape[1]
        height_factor = (anchors - 1) / (h - 1) if h > 1 else 1.0

        for i in range(anchors):
            # 对应 generate_target_regression 中的 row = rails_mask.shape[0] - 1 - i
            r_resized = anchors - 1 - i
            
            # 按照 resize_mask 的逆运算，将缩放后的锚点行映射回当前 cropped 图像的实际行坐标
            r_orig = round(r_resized / height_factor)
            
            if r_orig < 0 or r_orig >= h:
                continue
                
            xl_norm, xr_norm = path_gt[0, i], path_gt[1, i]
            
            # 归一化坐标转像素坐标
            pixel_xl = int(xl_norm * (w - 1))
            pixel_xr = int(xr_norm * (w - 1))
            
            if 0 <= pixel_xl < w:
                cv2.circle(vis_img, (pixel_xl, r_orig), 2, (0, 255, 0), -1)  # 绿点：左轨
            if 0 <= pixel_xr < w:
                cv2.circle(vis_img, (pixel_xr, r_orig), 2, (255, 0, 0), -1)  # 蓝点：右轨

        # 4. 保存图片
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_vis.jpg"), vis_img)

    # def generate_rails_mask(self, shape, annotation):
    #     rails_mask = Image.new("L", shape, 0)
    #     draw = ImageDraw.Draw(rails_mask)
    #     rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]
    #     for rail in rails:
    #         draw.line([tuple(xy) for xy in rail], fill=1, width=1)
    #     rails_mask = np.array(rails_mask)
    #     rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0
    #     for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
    #         rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0
    #     return rails_mask

    def generate_rails_mask(self, shape, annotation):
        rails_mask = Image.new("L", shape, 0)
        draw = ImageDraw.Draw(rails_mask)
        
        # 即使轨道贴边，也正常画线
        rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]
        for rail in rails:
            if len(rail) > 1:
                draw.line([tuple(xy) for xy in rail], fill=1, width=1)
                
        rails_mask = np.array(rails_mask)
        
        # 1. 裁剪起点上方多余部分
        valid_y_min = min(rails[0][:, 1].min(), rails[1][:, 1].min()) if len(rails[0]) > 0 and len(rails[1]) > 0 else 0
        rails_mask[: max(valid_y_min, 0), :] = 0
        
        # 2. 安全地处理每一行的多余像素（即使贴边也不崩塌）
        for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
            nonzero_indices = np.nonzero(rails_mask[row_idx, :])[0]
            if len(nonzero_indices) > 2:
                # 保留最左边一个，最右边一个，中间的清零
                # 这样即使贴边（nonzero_indices[0] == 0），也能安全保留
                left_most = nonzero_indices[0]
                right_most = nonzero_indices[-1]
                rails_mask[row_idx, :] = 0
                rails_mask[row_idx, left_most] = 1
                rails_mask[row_idx, right_most] = 1
                
        return rails_mask

    def random_crop(self, img, rails_mask):
        # extract sides rails coordinates (red rectangle in paper fig. 4)

      # FIXME: Jn 
        valid_rows = np.where(np.sum(rails_mask, axis=1) > 0)[0]
        if len(valid_rows) == 0:
            raise ValueError(f"No valid rail pixels found in the entire mask!")
        last_row_idx = valid_rows[-1]
        rails_mask_last_line_idx = np.nonzero(rails_mask[last_row_idx, :])[0]

        # rails_mask_last_line = rails_mask[-1, :]
        # rails_mask_last_line_idx = np.nonzero(rails_mask_last_line)[0]\

        most_left_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][0]
        most_right_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][-1]
        # center the crop around the rails (orange rectangle in paper fig. 4)
        base_margin_left = rails_mask_last_line_idx[0] - most_left_rail
        base_margin_right = most_right_rail - rails_mask_last_line_idx[-1]
        max_base_margin = max(base_margin_left, base_margin_right, 0)
        mean_crop_left = rails_mask_last_line_idx[0] - max_base_margin
        mean_crop_right = rails_mask_last_line_idx[-1] + max_base_margin


        # FIXME: Jn
        # 在 random_crop 计算出 mean_crop_left 和 mean_crop_right 后，加上边界保护：
        mean_crop_left = max(0, mean_crop_left)
        mean_crop_right = min(img.width - 1, mean_crop_right)

        # add sides margins (green rectangle in paper fig. 4)
        base_width = mean_crop_right - mean_crop_left + 1
        mean_crop_left -= base_width * self.config["crop_margin_sides"]
        mean_crop_right += base_width * self.config["crop_margin_sides"]
        # random left crop
        largest_margin = max(mean_crop_left, most_left_rail - mean_crop_left)
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_left = round(np.random.normal(mean_crop_left, std_dev))
        if random_crop_left > rails_mask_last_line_idx[0]:
            random_crop_left = 2 * rails_mask_last_line_idx[0] - random_crop_left
        random_crop_left = max(random_crop_left, 0)
        # random right crop
        largest_margin = max(
            mean_crop_right - most_right_rail, img.width - 1 - mean_crop_right
        )
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_right = round(np.random.normal(mean_crop_right, std_dev))
        if random_crop_right < rails_mask_last_line_idx[-1]:
            random_crop_right = 2 * rails_mask_last_line_idx[-1] - random_crop_right
        random_crop_right = min(random_crop_right, img.width - 1)
        # extract top rails coordinates (red rectangle in paper fig. 4)
        most_top_rail = np.nonzero(np.sum(rails_mask, axis=1))[0][0]
        # add top margin (green rectangle in paper fig. 4)
        rail_height = img.height - most_top_rail
        mean_crop_top = most_top_rail - rail_height * self.config["crop_margin_top"]
        # random top crop
        largest_margin = max(mean_crop_top, img.height - 1 - mean_crop_top)
        std_dev = largest_margin * self.config["std_dev_factor_top"]
        random_crop_top = round(np.random.normal(mean_crop_top, std_dev))
        random_crop_top = max(random_crop_top, 0)
        random_crop_top = min(random_crop_top, img.height - 2)  # at least 2 rows
        # crop image and mask
        img = img.crop(
            (random_crop_left, random_crop_top, random_crop_right + 1, img.height)
        )
        rails_mask = rails_mask[
            random_crop_top:, random_crop_left : random_crop_right + 1
        ]

        # FIXME: Jn
        # 【新增防护】如果裁剪后的宽度太小（比如小于 10 像素），强制赋一个最小宽度或重新处理
        MIN_WIDTH = 10
        if rails_mask.shape[1] < MIN_WIDTH:
            # 可以在这里做个简单的兜底，或者直接返回原图/扩大裁剪边界
            # 简单做法：通过 padding 或调整右边界来保证最小宽度
            pad_width = MIN_WIDTH - rails_mask.shape[1]
            random_crop_right += pad_width
            # 重新裁剪一次确保安全
            img = img.crop((random_crop_left, random_crop_top, random_crop_right + 1, img.height))
            rails_mask = rails_mask[random_crop_top:, random_crop_left : random_crop_right + 1]
        return img, rails_mask

    def resize_mask(self, mask, shape):
        # FIXME: Jn
        # 【新增双保险防护】防止高或宽小于等于 1 导致除以零
        if mask.shape[0] <= 1 or mask.shape[1] <= 1:
            # 返回一个合规形状的全零数组，不让进程崩掉
            return np.zeros(shape, dtype=np.uint8)
        height_factor = (shape[0] - 1) / (mask.shape[0] - 1)
        width_factor = (shape[1] - 1) / (mask.shape[1] - 1)
        resized_mask = np.zeros(shape, dtype=np.uint8)
        for i in range(resized_mask.shape[0]):
            row_mask = mask[round(i / height_factor), :]
            row_idx = np.nonzero(row_mask)[0]
            if len(row_idx) == 2:
                resized_mask[i, np.round(row_idx * width_factor).astype(int)] = 1
        return resized_mask

    def random_flip_lr(self, img, rails_mask):
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
            rails_mask = np.fliplr(rails_mask)
        return img, rails_mask

    # def generate_target_regression(self, rails_mask):
    #     unvalid_rows = np.where(np.sum(rails_mask, axis=1) != 2)[0]
    #     ylim_target = (
    #         float(1 - (unvalid_rows[-1] + 1) / rails_mask.shape[0])
    #         if len(unvalid_rows) > 0
    #         else 1.0
    #     )
    #     rails_mask = self.resize_mask(
    #         rails_mask, (self.config["anchors"], rails_mask.shape[1])
    #     )
    #     traj_target = np.array(
    #         [np.zeros(self.config["anchors"]), np.ones(self.config["anchors"])],
    #         dtype=np.float32,
    #     )
    #     for i in range(self.config["anchors"]):  # it's possible to vectorize this loop
    #         row = rails_mask.shape[0] - 1 - i
    #         rails_points = np.nonzero(rails_mask[row, :])[0]
    #         if len(rails_points) != 2:
    #             break
    #         rails_points_normalized = rails_points / (rails_mask.shape[1] - 1)
    #         traj_target[:, i] = rails_points_normalized
    #     return traj_target, ylim_target

    def generate_target_regression(self, rails_mask):
        # 1. 寻找所有有效行（即该行非零像素等于2的行，代表有左右轨道）
        valid_rows = np.where(np.sum(rails_mask, axis=1) == 2)[0]
        
        if len(valid_rows) > 0:
            # 轨道的最高起点（最上方的有效行索引）
            top_valid_row = valid_rows[0]
            # 计算从底向上的比例位置
            ylim_target = float(1.0 - top_valid_row / rails_mask.shape[0])
        else:
            ylim_target = 1.0

        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], rails_mask.shape[1])
        )
        
        traj_target = np.array(
            [np.zeros(self.config["anchors"]), np.ones(self.config["anchors"])],
            dtype=np.float32,
        )
        
        last_valid_left = 0.0
        last_valid_right = 1.0
        
        # 从底层往上扫描进行兜底，确保底部不留空隙
        for r_idx in range(rails_mask.shape[0] - 1, -1, -1):
            pts = np.nonzero(rails_mask[r_idx, :])[0]
            if len(pts) >= 2:
                last_valid_left = pts[0] / (rails_mask.shape[1] - 1)
                last_valid_right = pts[-1] / (rails_mask.shape[1] - 1)
                break

        for i in range(self.config["anchors"]):
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            
            if len(rails_points) >= 2:
                rails_points_normalized = np.array([rails_points[0], rails_points[-1]], dtype=np.float32) / (rails_mask.shape[1] - 1)
                traj_target[:, i] = rails_points_normalized
                last_valid_left = rails_points_normalized[0]
                last_valid_right = rails_points_normalized[1]
            else:
                traj_target[0, i] = last_valid_left
                traj_target[1, i] = last_valid_right
                
        return traj_target, ylim_target

    def generate_target_classification(self, rails_mask):
        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], self.config["classes"])
        )
        target = (
            np.ones((2, self.config["anchors"]), dtype=int) * self.config["classes"]
        )
        for i in range(self.config["anchors"]):
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            if len(rails_points) != 2:
                break
            target[:, i] = rails_points
        return target

    def generate_target_segmentation(self, rails_mask):
        target = np.zeros_like(rails_mask, dtype=np.uint8)
        row_indices, col_indices = np.nonzero(rails_mask)
        range_rows = np.arange(row_indices.min(), row_indices.max() + 1)
        for row in reversed(range_rows):
            rails_points = col_indices[row_indices == row]
            if len(rails_points) != 2:
                break
            target[row, rails_points[0] : rails_points[1] + 1] = 255
        return Image.fromarray(target)

    def get_perspective_weight_limit(self, percentile, logger):
        logger.info("\nCalculating perspective weight limit...")
        weights = []
        for i in range(len(self)):
            _, traj, ylim = self[i]
            rails = regression_to_rails(traj.numpy(), ylim.item())
            left_rail, right_rail = rails
            rail_width = right_rail[:, 0] - left_rail[:, 0]
            weight = 1 / rail_width
            weights += weight.tolist()
        limit = np.percentile(sorted(weights), percentile)
        logger.info(f"Perspective weight limit: {limit:.2f}")
        return limit
