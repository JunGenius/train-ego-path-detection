import torch
import torch.nn as nn
from configs.constants import FIXME_VERSION


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        target = target.flatten()
        prediction = prediction.view(target.shape[0], -1)
        return self.loss(prediction, target)


class TrainEgoPathRegressionLoss(nn.Module):

    def __init__(self, ylimit_loss_weight, perspective_weight_limit=None):
        super(TrainEgoPathRegressionLoss, self).__init__()
        self.ylimit_loss_weight = ylimit_loss_weight
        self.perspective_weight_limit = perspective_weight_limit
        self.unreduced_smae = nn.SmoothL1Loss(reduction="none", beta=0.005)
        self.batchaveraged_smae = nn.SmoothL1Loss(reduction="mean", beta=0.015)

    def trajectory_loss_v1(self, traj_prediction, traj_target, ylim_target):
        traj_se = self.unreduced_smae(traj_prediction, traj_target)  # (B, 2, H)
        ylim_target_idx = ylim_target * (traj_target.size(2) - 1)  # (B,)
        range_matrix = torch.arange(
            traj_target.size(2), device=ylim_target_idx.device
        ).expand(traj_target.size(0), -1)  # (B, H)
        loss_mask = (range_matrix <= ylim_target_idx.unsqueeze(1)).float()  # (B, H)
        rail_width = traj_target[:, 1, :] - traj_target[:, 0, :]  # (B, H)
        weights = (loss_mask / rail_width).unsqueeze(dim=1)  # (B, 1, H)
        if self.perspective_weight_limit is not None:
            weights = torch.clamp(weights, max=self.perspective_weight_limit)
        traj_loss = (traj_se * weights).sum(dim=(1, 2)) / loss_mask.sum(dim=1)
        mask = ylim_target == 0
        traj_loss[mask] = 0
        return traj_loss.mean()  # ()

    def trajectory_loss_v2(self, traj_prediction, traj_target):
        # traj_prediction / traj_target: (B, 2, H)
        # 所有锚点都参与监督，不再用 ylim 截断
        traj_se = self.unreduced_smae(traj_prediction, traj_target)  # (B, 2, H)

        rail_width = traj_target[:, 1, :] - traj_target[:, 0, :]  # (B, H)
        safe_width = torch.clamp(rail_width, min=1e-6)
        weights = (1.0 / safe_width).unsqueeze(dim=1)  # (B, 1, H)

        if self.perspective_weight_limit is not None:
            weights = torch.clamp(weights, max=self.perspective_weight_limit)

        # 对每个样本的 (2, H) 求均值，再对 batch 求均值
        traj_loss = (traj_se * weights).mean(dim=(1, 2))
        return traj_loss.mean()
    
    def ylim_loss_v1(self, ylim_prediction, ylim_target):
        return self.batchaveraged_smae(torch.sigmoid(ylim_prediction), ylim_target)

    def ylim_loss_v2(self, ylim_prediction, ylim_target):
        # ylim_prediction: (B, 2) logits -> sigmoid 后映射到 [0, 1]
        # ylim_target: (B, 2) -> [y_start, y_end]
        ylim_clamped = torch.clamp(ylim_target, min=0.0, max=1.0)
        y_start_gt = torch.minimum(ylim_clamped[:, 0], ylim_clamped[:, 1])
        y_end_gt = torch.maximum(ylim_clamped[:, 0], ylim_clamped[:, 1])
        target_range = torch.stack([y_start_gt, y_end_gt], dim=-1)

        pred_range = torch.sigmoid(ylim_prediction)
        return self.batchaveraged_smae(pred_range, target_range)

    def forward(self, prediction, target):
        if FIXME_VERSION:
            traj_target, ylim_target = target
            traj_prediction = prediction[:, :-2].view_as(traj_target)
            ylim_prediction = prediction[:, -2:]

            traj_loss = self.trajectory_loss_v2(traj_prediction, traj_target)
            ylim_loss = self.ylim_loss_v2(ylim_prediction, ylim_target)
            return traj_loss + self.ylimit_loss_weight * ylim_loss
        else:
            traj_target, ylim_target = target
            traj_prediction = prediction[:, :-1].view_as(traj_target)
            ylim_prediction = prediction[:, -1]
            traj_loss = self.trajectory_loss_v1(traj_prediction, traj_target, ylim_target)
            ylim_loss = self.ylim_loss_v1(ylim_prediction, ylim_target)
            combined_loss = traj_loss + self.ylimit_loss_weight * ylim_loss
            return combined_loss

# FIXME: Jn 修改 适配 ylim [start,stop] 二维输出
# class TrainEgoPathRegressionLoss(nn.Module):
#     def __init__(self, ylimit_loss_weight, perspective_weight_limit=None):
#         super().__init__()
#         self.ylimit_loss_weight = ylimit_loss_weight
#         self.perspective_weight_limit = perspective_weight_limit
#         self.unreduced_smae = nn.SmoothL1Loss(reduction="none", beta=0.005)
#         self.batchaveraged_smae = nn.SmoothL1Loss(reduction="mean", beta=0.015)

#     def trajectory_loss(self, traj_prediction, traj_target):
#         # traj_prediction / traj_target: (B, 2, H)
#         # 所有锚点都参与监督，不再用 ylim 截断
#         traj_se = self.unreduced_smae(traj_prediction, traj_target)  # (B, 2, H)

#         rail_width = traj_target[:, 1, :] - traj_target[:, 0, :]  # (B, H)
#         safe_width = torch.clamp(rail_width, min=1e-6)
#         weights = (1.0 / safe_width).unsqueeze(dim=1)  # (B, 1, H)

#         if self.perspective_weight_limit is not None:
#             weights = torch.clamp(weights, max=self.perspective_weight_limit)

#         # 对每个样本的 (2, H) 求均值，再对 batch 求均值
#         traj_loss = (traj_se * weights).mean(dim=(1, 2))
#         return traj_loss.mean()

#     def ylim_loss(self, ylim_prediction, ylim_target):
#         # ylim_prediction: (B, 2) logits -> sigmoid 后映射到 [0, 1]
#         # ylim_target: (B, 2) -> [y_start, y_end]
#         ylim_clamped = torch.clamp(ylim_target, min=0.0, max=1.0)
#         y_start_gt = torch.minimum(ylim_clamped[:, 0], ylim_clamped[:, 1])
#         y_end_gt = torch.maximum(ylim_clamped[:, 0], ylim_clamped[:, 1])
#         target_range = torch.stack([y_start_gt, y_end_gt], dim=-1)

#         pred_range = torch.sigmoid(ylim_prediction)
#         return self.batchaveraged_smae(pred_range, target_range)

#     def forward(self, prediction, target):
#         traj_target, ylim_target = target
#         traj_prediction = prediction[:, :-2].view_as(traj_target)
#         ylim_prediction = prediction[:, -2:]

#         traj_loss = self.trajectory_loss(traj_prediction, traj_target)
#         ylim_loss = self.ylim_loss(ylim_prediction, ylim_target)
#         return traj_loss + self.ylimit_loss_weight * ylim_loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        prediction = prediction.flatten(start_dim=1)  # (B, H * W)
        target = target.flatten(start_dim=1)
        zero_target_mask = target.sum(dim=1) == 0  # (B,)
        if zero_target_mask.any():
            prediction[zero_target_mask] = 1 - prediction[zero_target_mask]
            target[zero_target_mask] = 1 - target[zero_target_mask]
        intersection = (prediction * target).sum(dim=1)  # (B,)
        cardinality = (prediction + target).sum(dim=1)
        scores = 2 * intersection / cardinality
        return 1 - scores.mean()  # ()
