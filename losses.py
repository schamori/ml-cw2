import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDiceScore(nn.Module):
    def __init__(self, weight_matrix, epsilon=1e-6):
        super().__init__()
        # self.register_buffer("weight_matrix", weight_matrix)
        self.weight_matrix = weight_matrix
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Probabilities (B, C, D, H, W) or (B, C, H, W) - Output of Softmax
            target (torch.Tensor): Ground Truth Indices (B, D, H, W) or (B, H, W)
        """
        num_classes = pred.shape[1]

        # 1. Convert Target to One-Hot
        # For 3D: (B, D, H, W) -> (B, C, D, H, W)
        # For 2D: (B, H, W) -> (B, C, H, W)
        target_onehot = F.one_hot(target, num_classes)
        # Move class dimension to position 1
        if target.dim() == 3:  # 2D case: (B, H, W)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:  # 3D case: (B, D, H, W)
            target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        # 2. Flatten spatial dimensions for efficient matrix multiplication
        # Shapes become: (B, C, N) where N = D*H*W or H*W
        pred_flat = pred.flatten(2)
        target_flat = target_onehot.flatten(2)

        # 3. Compute Soft Confusion Matrix via Einsum
        # Equation: For every batch b, sum over pixels n: pred(c) * target(k)
        # Result: (C, C) matrix where row=Pred, col=Target
        # This effectively sums up the "probability mass" for every pred/target pair
        soft_cm = torch.einsum("bcn, bkn -> ck", pred_flat, target_flat)

        # 4. Apply Weights
        # weight_matrix is (C, C). Element-wise multiplication applies penalties.
        # Since diagonal of weight_matrix is 0, TP contributions are zeroed out here.
        weighted_cm = soft_cm * self.weight_matrix

        # 5. Calculate Components

        # TP: Diagonal of the original Soft Confusion Matrix (unweighted)
        tp = torch.diagonal(soft_cm)

        # FP_weighted: Sum of Weighted Rows (Predicted c, Actual k)
        # sum(dim=1) collapses columns
        fp_weighted = weighted_cm.sum(dim=1)

        # FN_weighted: Sum of Weighted Columns (Actual c, Predicted k)
        # sum(dim=0) collapses rows
        fn_weighted = weighted_cm.sum(dim=0)

        # 6. Dice Formula
        numerator = 2 * tp
        denominator = (2 * tp) + fp_weighted + fn_weighted + self.epsilon

        scores = numerator / denominator

        return scores


class DiceCELoss(nn.Module):
    """Combined Dice Loss + Cross Entropy Loss with Weighted Dice"""

    def __init__(self, num_classes, weight_matrix, ce_weight=1.0, dice_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.weighted_dice_score = WeightedDiceScore(weight_matrix)

    def dice_loss(self, pred, target):
        """
        Calculate Weighted Dice loss for all classes
        pred: (B, C, D, H, W) or (B, C, H, W) - raw logits
        target: (B, D, H, W) or (B, H, W) - class indices
        """
        # Convert logits to probabilities
        pred_softmax = F.softmax(pred, dim=1)

        # Get weighted dice scores per class
        dice_scores = self.weighted_dice_score(pred_softmax, target)

        # Average dice loss (1 - dice score)
        dice_loss = 1.0 - dice_scores.mean()

        return dice_loss

    def forward(self, pred, target):
        """
        pred: (B, C, D, H, W) or (B, C, H, W)
        target: (B, D, H, W) or (B, H, W)
        """
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss, ce_loss, dice_loss
