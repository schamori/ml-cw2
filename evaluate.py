"""
Evaluation script to compute metrics for predictions against ground truth.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from pathlib import Path
import SimpleITK as sitk

from dataset import LABELS

# =============================================================================
# CONFIGURATION - Specify your prediction directory here
# =============================================================================
PRED_DIR = "predictions/training_new_non_weighted"
# =============================================================================

DATA_DIR = Path("./7013610/data/data")


def compute_dice_per_class(pred, gt, num_classes):
    """Compute Dice score for each class."""
    dice_scores = {}
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.sum(pred_cls & gt_cls)
        union = np.sum(pred_cls) + np.sum(gt_cls)

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / union

        dice_scores[cls] = dice

    return dice_scores


def load_nifti(path):
    """Load NIfTI file and return numpy array."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img)


def main():
    print("=" * 80)
    print("Prediction Evaluation")
    print("=" * 80)

    pred_dir = Path(PRED_DIR)

    if not pred_dir.exists():
        print(f"Error: Prediction directory not found: {PRED_DIR}")
        return

    print(f"\nPrediction directory: {PRED_DIR}")

    # Find all prediction files
    pred_files = sorted(pred_dir.glob("*_pred.nii.gz"))
    print(f"Found {len(pred_files)} prediction files")

    if len(pred_files) == 0:
        print("Error: No prediction files found!")
        return

    # Find corresponding ground truth masks
    num_classes = len(LABELS)
    all_dice_scores = {cls: [] for cls in range(num_classes)}

    results = []

    for pred_path in pred_files:
        # Get corresponding mask file
        base_name = pred_path.stem.replace('_pred', '_mask')
        mask_path = DATA_DIR / f"{base_name}.nii.gz"

        if not mask_path.exists():
            print(f"Warning: Ground truth not found for {pred_path.name}, skipping...")
            continue

        # Load prediction and ground truth
        pred = load_nifti(pred_path)
        gt = load_nifti(mask_path)

        # Compute dice scores per class
        dice_scores = compute_dice_per_class(pred, gt, num_classes)

        for cls, dice in dice_scores.items():
            all_dice_scores[cls].append(dice)

        # Mean dice (excluding background)
        mean_dice = np.mean([dice_scores[c] for c in range(1, num_classes)])

        results.append({
            'file': pred_path.name,
            'mean_dice': mean_dice,
            'per_class': dice_scores
        })

    # Print per-file results
    print("\n" + "-" * 80)
    print("Per-file Results (Mean Dice excluding background)")
    print("-" * 80)
    print(f"{'File':<50} {'Mean Dice':<10}")
    print("-" * 80)

    for result in sorted(results, key=lambda x: x['mean_dice'], reverse=True):
        print(f"{result['file']:<50} {result['mean_dice']:.4f}")

    # Print per-class summary
    print("\n" + "=" * 80)
    print("Per-Class Dice Scores (averaged over all samples)")
    print("=" * 80)
    print(f"{'Class':<5} {'Label':<25} {'Mean Dice':<12} {'Std':<10}")
    print("-" * 80)

    class_means = []
    for cls in range(num_classes):
        if len(all_dice_scores[cls]) > 0:
            mean_dice = np.mean(all_dice_scores[cls])
            std_dice = np.std(all_dice_scores[cls])
            label_name = LABELS.get(cls, f"Class {cls}")
            print(f"{cls:<5} {label_name:<25} {mean_dice:.4f}       {std_dice:.4f}")
            if cls > 0:  # Exclude background from overall mean
                class_means.append(mean_dice)

    # Overall summary
    print("-" * 80)
    print(f"\nOverall Mean Dice (excluding background): {np.mean(class_means):.4f}")
    print(f"Overall Std Dice (excluding background):  {np.std(class_means):.4f}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
