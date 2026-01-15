"""
Evaluation script to compare predictions from two model checkpoints on the test set.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk

from simple_nnunet_training import build_nnunet_network, prepare_data_lists
from dataset import MedicalImageDataset, collate_fn_single, LABELS, TARGET_SPACING

# =============================================================================
# CONFIGURATION - Specify your two checkpoint paths here
# =============================================================================
WEIGHT_1 = "checkpoints/training_new_non_weighted.pth"
WEIGHT_2 = "checkpoints/training_new_weighted.pth"
# =============================================================================

DATA_DIR = Path("./7013610/data/data")
OUTPUT_DIR = Path("./predictions")


def compute_dice(pred1, pred2, smooth=1e-5):
    """Compute Dice score between two prediction masks."""
    intersection = (pred1 == pred2).sum()
    total = pred1.numel() + pred2.numel()
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.item()


def load_model(checkpoint_path, device, num_classes=9):
    """Load model from checkpoint."""
    model = build_nnunet_network(num_input_channels=1, num_classes=num_classes)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def run_inference(model, dataset, device):
    """Run inference on dataset and return predictions with file info."""
    predictions = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Running inference"):
            image, mask, original_shape = dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dim

            output = model(image)
            if isinstance(output, (list, tuple)):
                output = output[0]

            pred = torch.argmax(output, dim=1).squeeze(0)  # Remove batch dim

            # Unpad to original shape
            d, h, w = original_shape
            pred_unpadded = pred[:d, :h, :w].cpu()

            predictions.append({
                'prediction': pred_unpadded,
                'original_shape': original_shape,
                'file_path': dataset.image_files[idx]
            })

    return predictions


def save_predictions(predictions, output_dir, original_images):
    """Save predictions as NIfTI files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for pred_info, img_path in zip(predictions, original_images):
        # Load original image to get metadata
        original_sitk = sitk.ReadImage(str(img_path))

        # Create prediction image
        pred_np = pred_info['prediction'].numpy().astype(np.uint8)
        pred_sitk = sitk.GetImageFromArray(pred_np)
        pred_sitk.SetSpacing(TARGET_SPACING)

        # Save with same base name
        base_name = img_path.stem.replace('_img', '_pred')
        save_path = output_dir / f"{base_name}.nii.gz"
        sitk.WriteImage(pred_sitk, str(save_path))

    print(f"Saved {len(predictions)} predictions to {output_dir}")


def main():
    print("=" * 80)
    print("Model Comparison Evaluation")
    print("=" * 80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check weights exist
    weight1_path = Path(WEIGHT_1)
    weight2_path = Path(WEIGHT_2)

    if not weight1_path.exists():
        print(f"Error: Weight file not found: {WEIGHT_1}")
        return
    if not weight2_path.exists():
        print(f"Error: Weight file not found: {WEIGHT_2}")
        return

    print(f"\nWeight 1: {WEIGHT_1}")
    print(f"Weight 2: {WEIGHT_2}")

    # Get test set
    print("\nLoading test set...")
    _, _, _, _, test_images, test_masks = prepare_data_lists(
        DATA_DIR, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"Test samples: {len(test_images)}")

    if len(test_images) == 0:
        print("Error: No test images found!")
        return

    # Create test dataset
    test_dataset = MedicalImageDataset(test_images, test_masks, augment=False)

    # Load models
    print("\nLoading models...")
    num_classes = len(LABELS)
    model1 = load_model(weight1_path, device, num_classes)
    model2 = load_model(weight2_path, device, num_classes)

    # Run inference
    print("\n" + "-" * 40)
    print(f"Model 1: {weight1_path.stem}")
    print("-" * 40)
    predictions1 = run_inference(model1, test_dataset, device)

    print("\n" + "-" * 40)
    print(f"Model 2: {weight2_path.stem}")
    print("-" * 40)
    predictions2 = run_inference(model2, test_dataset, device)

    # Create output directories based on weight names
    output_dir1 = OUTPUT_DIR / weight1_path.stem
    output_dir2 = OUTPUT_DIR / weight2_path.stem

    # Save predictions
    print("\nSaving predictions...")
    save_predictions(predictions1, output_dir1, test_images)
    save_predictions(predictions2, output_dir2, test_images)

    # Compare predictions - find files where predictions differ most
    print("\n" + "=" * 80)
    print("Comparing Predictions")
    print("=" * 80)

    comparison_results = []

    for p1, p2 in zip(predictions1, predictions2):
        pred1 = p1['prediction']
        pred2 = p2['prediction']
        file_path = p1['file_path']

        # Compute dice between the two predictions
        dice = compute_dice(pred1, pred2)

        comparison_results.append({
            'file': file_path.name,
            'dice': dice
        })

    # Sort by dice (ascending = most different first)
    comparison_results.sort(key=lambda x: x['dice'])

    # Print top 10 most different
    print("\nTop 10 files where predictions differ most (lowest dice):")
    print("-" * 60)
    print(f"{'Rank':<6} {'File':<40} {'Dice':<10}")
    print("-" * 60)

    for i, result in enumerate(comparison_results[:10], 1):
        print(f"{i:<6} {result['file']:<40} {result['dice']:.4f}")

    print("-" * 60)

    # Summary stats
    all_dices = [r['dice'] for r in comparison_results]
    print(f"\nOverall prediction similarity:")
    print(f"  Mean Dice:   {np.mean(all_dices):.4f}")
    print(f"  Min Dice:    {np.min(all_dices):.4f}")
    print(f"  Max Dice:    {np.max(all_dices):.4f}")
    print(f"  Std Dice:    {np.std(all_dices):.4f}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Predictions saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
