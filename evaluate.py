"""
Script to generate predictions from a model checkpoint.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from pathlib import Path
import SimpleITK as sitk

from experiment import build_network, prepare_data_lists
from dataset import MedicalImageDataset, LABELS, TARGET_SPACING


WEIGHT_PATH = "checkpoints/best_model.pth"
# =============================================================================

DATA_DIR = Path("./7013610/data/data")
OUTPUT_DIR = Path("./predictions")


def load_model(checkpoint_path, device, num_classes=9):
    """Load model from checkpoint."""
    model = build_network(num_input_channels=1, num_classes=num_classes)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def run_inference(model, dataset, device):
    """Run inference on dataset and return predictions with file info."""
    predictions = []

    with torch.no_grad():
        for idx in range(len(dataset)):
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


def save_predictions(predictions, output_dir):
    """Save predictions as NIfTI files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for pred_info in predictions:
        pred_np = pred_info['prediction'].numpy().astype(np.uint8)
        pred_sitk = sitk.GetImageFromArray(pred_np)
        pred_sitk.SetSpacing(TARGET_SPACING)

        # Save with same base name
        base_name = pred_info['file_path'].stem.replace('_img', '_pred')
        save_path = output_dir / f"{base_name}.nii.gz"
        sitk.WriteImage(pred_sitk, str(save_path))

    print(f"Saved {len(predictions)} predictions to {output_dir}")


def main():
    print("=" * 80)
    print("Generating Predictions")
    print("=" * 80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check weight exists
    weight_path = Path(WEIGHT_PATH)

    if not weight_path.exists():
        print(f"Error: Weight file not found: {WEIGHT_PATH}")
        return

    print(f"\nWeight: {WEIGHT_PATH}")

    # Get test set
    print("\nLoading test set...")
    _, _, _, _, test_images, test_masks = prepare_data_lists(
        DATA_DIR, train_ratio=0.7, val_ratio=0.15, _test_ratio=0.15
    )
    print(f"Test samples: {len(test_images)}")

    if len(test_images) == 0:
        print("Error: No test images found!")
        return

    # Create test dataset
    test_dataset = MedicalImageDataset(test_images, test_masks, augment=False)

    # Load model
    print("\nLoading model...")
    num_classes = len(LABELS)
    model = load_model(weight_path, device, num_classes)

    # Run inference
    print("\nRunning inference...")
    predictions = run_inference(model, test_dataset, device)

    # Save predictions
    output_dir = OUTPUT_DIR / weight_path.stem
    save_predictions(predictions, output_dir)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
