"""
Evaluate predictions from a model on the test set.
Uses the same seed (42) and split ratios as training to identify test samples.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from pathlib import Path
import SimpleITK as sitk

from dataset import LABELS, TARGET_SPACING, resample_image
from losses import compute_weighted_dice_score
import weighted_matrices

# Seed and split ratios (same as training)
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
DATA_DIR = Path("./7013610/data/data")
PRED_DIR = Path("./predictions/training_new_non_weighted")

# Foreground labels (excluding background)
FOREGROUND_LABELS = {k: v for k, v in LABELS.items() if k != "background"}


def prepare_data_lists(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare lists of training, validation, and test files
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    data_dir = Path(data_dir)

    # Find all image files
    image_files = sorted(data_dir.glob("*_img.nii*"))

    all_images = []
    all_masks = []

    for img_file in image_files:
        # Construct corresponding mask file
        mask_file = img_file.parent / img_file.name.replace('_img', '_mask')

        if not mask_file.exists():
            print(f"Warning: Mask not found for {img_file}, skipping...")
            continue

        all_images.append(img_file)
        all_masks.append(mask_file)

    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(all_images))

    # Calculate split indices
    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Split data
    train_images = [all_images[i] for i in train_indices]
    train_masks = [all_masks[i] for i in train_indices]

    val_images = [all_images[i] for i in val_indices]
    val_masks = [all_masks[i] for i in val_indices]

    test_images = [all_images[i] for i in test_indices]
    test_masks = [all_masks[i] for i in test_indices]

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def calculate_dice_per_class(pred, target, num_classes, smooth=1e-5, include_background=False):
    """
    Calculate Dice score for each class (excluding background by default)

    pred: (D, H, W) - predicted class indices
    target: (D, H, W) - ground truth class indices
    include_background: if False, excludes class 0 (background) from calculation
    """
    dice_scores = {}
    label_names = {int(v): k for k, v in LABELS.items()}

    start_class = 0 if include_background else 1

    for class_idx in range(start_class, num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        if union > 0:
            dice_score = (2.0 * intersection + smooth) / (union + smooth)
        else:
            dice_score = torch.tensor(float('nan'))

        class_name = label_names.get(class_idx, f"class_{class_idx}")
        dice_scores[class_name] = dice_score.item()

    return dice_scores


def load_mask(mask_path, target_spacing=TARGET_SPACING):
    """Load and resample a mask file."""
    mask_sitk = sitk.ReadImage(str(mask_path))
    mask_sitk = resample_image(mask_sitk, target_spacing, is_mask=True)
    mask = sitk.GetArrayFromImage(mask_sitk).astype(np.int64)
    return torch.from_numpy(mask)


def load_prediction(pred_path):
    """Load a prediction file."""
    pred_sitk = sitk.ReadImage(str(pred_path))
    pred = sitk.GetArrayFromImage(pred_sitk).astype(np.int64)
    return torch.from_numpy(pred)


def evaluate_predictions(pred_dir, test_masks, weight_matrix, device):
    """
    Evaluate predictions in a directory against ground truth masks.

    Returns:
        dict with per-class dice scores and weighted dice scores
    """
    num_classes = len(LABELS)
    all_dice_scores = {k: [] for k in FOREGROUND_LABELS.keys()}
    all_weighted_dice_scores = []

    for mask_path in test_masks:
        # Get prediction file path
        # Mask: XXXXXX_mask.nii.gz -> Prediction: XXXXXX_pred.nii.gz
        base_name = mask_path.name.replace('_mask', '_pred')
        # Handle both .nii and .nii.gz extensions
        if base_name.endswith('.nii'):
            base_name = base_name + '.gz'
        elif not base_name.endswith('.nii.gz'):
            base_name = base_name.replace('.nii.gz', '') + '_pred.nii.gz'

        pred_path = pred_dir / base_name

        if not pred_path.exists():
            # Try alternative naming
            stem = mask_path.stem.replace('_mask', '')
            if stem.endswith('.nii'):
                stem = stem[:-4]
            pred_path = pred_dir / f"{stem}_pred.nii.gz"

        if not pred_path.exists():
            print(f"Warning: Prediction not found for {mask_path.name}")
            continue

        # Load ground truth and prediction
        mask = load_mask(mask_path)
        pred = load_prediction(pred_path)

        # Ensure same shape (crop to minimum if needed)
        min_d = min(mask.shape[0], pred.shape[0])
        min_h = min(mask.shape[1], pred.shape[1])
        min_w = min(mask.shape[2], pred.shape[2])

        mask = mask[:min_d, :min_h, :min_w]
        pred = pred[:min_d, :min_h, :min_w]

        # Calculate dice scores per class
        dice_scores = calculate_dice_per_class(pred, mask, num_classes, include_background=False)

        for class_name, score in dice_scores.items():
            if not np.isnan(score):
                all_dice_scores[class_name].append(score)

        # Calculate weighted dice score
        weighted_scores = compute_weighted_dice_score(
            pred.to(device),
            mask.to(device),
            weight_matrix,
            num_classes=num_classes
        )
        all_weighted_dice_scores.append(weighted_scores.cpu())

    # Calculate mean dice scores per class
    mean_dice_scores = {}
    for class_name, scores in all_dice_scores.items():
        if len(scores) > 0:
            mean_dice_scores[class_name] = np.mean(scores)
        else:
            mean_dice_scores[class_name] = float('nan')

    # Calculate mean foreground dice
    valid_scores = [s for s in mean_dice_scores.values() if not np.isnan(s)]
    mean_foreground_dice = np.mean(valid_scores) if valid_scores else float('nan')

    # Calculate mean and std of weighted dice scores
    if all_weighted_dice_scores:
        stacked_weighted_dice = torch.stack(all_weighted_dice_scores)
        mean_weighted_dice = stacked_weighted_dice.mean(dim=0)
        std_weighted_dice = stacked_weighted_dice.std(dim=0)
        mean_weighted_dice_dict = {
            f"{list(LABELS.keys())[i]}": mean_weighted_dice[i].item()
            for i in range(num_classes)
        }
        std_weighted_dice_dict = {
            f"{list(LABELS.keys())[i]}": std_weighted_dice[i].item()
            for i in range(num_classes)
        }
        mean_weighted_foreground_dice = mean_weighted_dice[1:].mean().item()
        # Std of foreground weighted dice (across all samples and foreground classes)
        std_weighted_foreground_dice = stacked_weighted_dice[:, 1:].std().item()
    else:
        mean_weighted_dice_dict = {}
        std_weighted_dice_dict = {}
        mean_weighted_foreground_dice = float('nan')
        std_weighted_foreground_dice = float('nan')

    return {
        'dice_per_class': mean_dice_scores,
        'mean_foreground_dice': mean_foreground_dice,
        'weighted_dice_per_class': mean_weighted_dice_dict,
        'std_weighted_dice_per_class': std_weighted_dice_dict,
        'mean_weighted_foreground_dice': mean_weighted_foreground_dice,
        'std_weighted_foreground_dice': std_weighted_foreground_dice,
        'num_samples': len(all_weighted_dice_scores)
    }


def print_results(name, results):
    """Print evaluation results in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"Results: {name}")
    print(f"{'=' * 60}")
    print(f"Number of samples evaluated: {results['num_samples']}")

    print(f"\nDice Scores per Class (excluding background):")
    print("-" * 50)
    for class_name, score in results['dice_per_class'].items():
        if not np.isnan(score):
            print(f"  {class_name:35s}: {score:.4f}")
        else:
            print(f"  {class_name:35s}: N/A")
    print("-" * 50)
    print(f"  {'Mean Foreground Dice':35s}: {results['mean_foreground_dice']:.4f}")

    print(f"\nWeighted Dice Scores per Class (mean ± std):")
    print("-" * 60)
    for class_name, score in results['weighted_dice_per_class'].items():
        std = results['std_weighted_dice_per_class'].get(class_name, float('nan'))
        print(f"  {class_name:35s}: {score:.4f} ± {std:.4f}")
    print("-" * 60)
    print(f"  {'Mean Weighted Foreground Dice':35s}: {results['mean_weighted_foreground_dice']:.4f} ± {results['std_weighted_foreground_dice']:.4f}")


def main():
    print("=" * 70)
    print("Prediction Evaluation")
    print("=" * 70)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check prediction directory exists
    if not PRED_DIR.exists():
        print(f"Error: Prediction directory not found: {PRED_DIR}")
        return

    print(f"\nPrediction directory: {PRED_DIR}")

    # Get test set using same seed and ratios as training
    print(f"\nLoading test set (seed={SEED}, split={TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO})...")

    try:
        _, _, _, _, test_images, test_masks = prepare_data_lists(
            DATA_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO
        )
        print(f"Test samples from data split: {len(test_masks)}")
    except Exception as e:
        print(f"Could not load data split from {DATA_DIR}: {e}")
        print("Falling back to using all predictions as test set...")

        # Fallback: use prediction files to infer test set
        pred_files = sorted(PRED_DIR.glob("*_pred.nii.gz"))
        test_masks = []
        for pred_file in pred_files:
            # Convert prediction name to mask name
            base = pred_file.stem.replace('_pred', '')
            if base.endswith('.nii'):
                base = base[:-4]
            mask_path = DATA_DIR / f"{base}_mask.nii.gz"
            test_masks.append(mask_path)
        print(f"Inferred test samples from predictions: {len(test_masks)}")

    if len(test_masks) == 0:
        print("Error: No test samples found!")
        return

    # Create weight matrix (same as training)
    sens_matrix = weighted_matrices.create_sens_matrix(weighted_matrices.SENSITIVITY_RANKINGS)
    dist_matrix = weighted_matrices.create_dist_matrix(weighted_matrices.AVG_DISTANCES)
    importance = 0.9
    weight_matrix = weighted_matrices.create_weighted_matrix(
        sens_matrix=sens_matrix,
        dist_matrix=dist_matrix,
        sens_importance=importance
    )
    weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32).to(device)

    # Evaluate predictions
    print("\n" + "-" * 70)
    results = evaluate_predictions(
        PRED_DIR,
        test_masks,
        weight_matrix,
        device
    )
    print_results("Model", results)

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
