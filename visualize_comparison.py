import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import matplotlib.patches as mpatches

LABELS = {
    0: "background",
    1: "urinary_bladder",
    2: "bone_hips",
    3: "obturator_internus",
    4: "transition_zone_prostate",
    5: "central_zone_prostate",
    6: "rectum",
    7: "seminal_vesicles",
    8: "neurovascular_bundle",
}

# Define colors for each class (using a colormap)
COLORS = {
    0: [0, 0, 0],           # background - black
    1: [1, 0, 0],           # urinary_bladder - red
    2: [0, 1, 0],           # bone_hips - green
    3: [0, 0, 1],           # obturator_internus - blue
    4: [1, 1, 0],           # transition_zone_prostate - yellow
    5: [1, 0, 1],           # central_zone_prostate - magenta
    6: [0, 1, 1],           # rectum - cyan
    7: [1, 0.5, 0],         # seminal_vesicles - orange
    8: [0.5, 0, 0.5],       # neurovascular_bundle - purple
}


def create_color_overlay(image, mask, alpha=0.5):
    """Create a colored overlay of the mask on the image"""
    # Normalize image to 0-1
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Create RGB image
    rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)

    # Apply colored mask
    for class_id in range(1, len(LABELS)):  # Skip background
        mask_class = (mask == class_id)
        if mask_class.any():
            color = COLORS[class_id]
            for c in range(3):
                rgb[:, :, c] = np.where(mask_class,
                                       alpha * color[c] + (1 - alpha) * rgb[:, :, c],
                                       rgb[:, :, c])

    return rgb


def visualize_comparison(image_path, gt_path, pred_path, num_slices=6):
    """Visualize comparison between ground truth and prediction"""

    print("Loading data...")
    # Load data
    img_sitk = sitk.ReadImage(str(image_path))
    image = sitk.GetArrayFromImage(img_sitk)  # Shape: (Z, Y, X)

    gt_sitk = sitk.ReadImage(str(gt_path))
    gt_mask = sitk.GetArrayFromImage(gt_sitk)

    pred_sitk = sitk.ReadImage(str(pred_path))
    pred_mask = sitk.GetArrayFromImage(pred_sitk)

    print(f"Image shape: {image.shape}")
    print(f"Ground truth shape: {gt_mask.shape}")
    print(f"Prediction shape: {pred_mask.shape}")

    # Select slices evenly distributed through the volume
    z_max = image.shape[0]
    slice_indices = np.linspace(z_max // 6, z_max - z_max // 6, num_slices, dtype=int)

    # Create figure
    fig, axes = plt.subplots(num_slices, 4, figsize=(16, 4 * num_slices))

    if num_slices == 1:
        axes = axes.reshape(1, -1)

    for i, slice_idx in enumerate(slice_indices):
        # Get slices
        img_slice = image[slice_idx, :, :]
        gt_slice = gt_mask[slice_idx, :, :]
        pred_slice = pred_mask[slice_idx, :, :]

        # Create overlays
        gt_overlay = create_color_overlay(img_slice, gt_slice, alpha=0.6)
        pred_overlay = create_color_overlay(img_slice, pred_slice, alpha=0.6)

        # Plot original image
        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f'Slice {slice_idx}: Original Image')
        axes[i, 0].axis('off')

        # Plot ground truth mask
        axes[i, 1].imshow(gt_slice, cmap='tab10', vmin=0, vmax=len(LABELS)-1)
        axes[i, 1].set_title(f'Ground Truth Mask')
        axes[i, 1].axis('off')

        # Plot prediction mask
        axes[i, 2].imshow(pred_slice, cmap='tab10', vmin=0, vmax=len(LABELS)-1)
        axes[i, 2].set_title(f'Prediction (Random Weights)')
        axes[i, 2].axis('off')

        # Plot overlay comparison
        axes[i, 3].imshow(img_slice, cmap='gray')
        axes[i, 3].imshow(gt_slice, cmap='Reds', alpha=0.3, vmin=0, vmax=len(LABELS)-1)
        axes[i, 3].imshow(pred_slice, cmap='Blues', alpha=0.3, vmin=0, vmax=len(LABELS)-1)
        axes[i, 3].set_title(f'Overlay: GT(Red) vs Pred(Blue)')
        axes[i, 3].axis('off')

    # Create legend
    legend_elements = []
    for class_id, label_name in LABELS.items():
        if class_id == 0:
            continue
        color = COLORS[class_id]
        legend_elements.append(mpatches.Patch(color=color, label=f'{class_id}: {label_name}'))

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    # Save figure
    output_path = Path(image_path).parent / "comparison_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("Class-wise comparison:")
    print("="*60)

    for class_id, label_name in LABELS.items():
        gt_count = np.sum(gt_mask == class_id)
        pred_count = np.sum(pred_mask == class_id)
        gt_pct = (gt_count / gt_mask.size) * 100
        pred_pct = (pred_count / pred_mask.size) * 100

        print(f"\nClass {class_id} ({label_name}):")
        print(f"  Ground Truth: {gt_count:6d} voxels ({gt_pct:5.2f}%)")
        print(f"  Prediction:   {pred_count:6d} voxels ({pred_pct:5.2f}%)")
        print(f"  Difference:   {pred_count - gt_count:+6d} voxels ({pred_pct - gt_pct:+5.2f}%)")


if __name__ == "__main__":
    data_dir = Path(r"./7013610/data/data")

    image_path = data_dir / "001000_img.nii"
    gt_path = data_dir / "001000_mask.nii"
    pred_path = data_dir / "001000_pred_nnunet.nii.gz"

    visualize_comparison(image_path, gt_path, pred_path, num_slices=6)
