"""
3D Medical Image Dataset with Elastic/B-spline Deformation Augmentations
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# Target spacing for resampling (median of dataset)
TARGET_SPACING = (1.0, 1.0, 2.0)

# Labels definition
LABELS = {
    "background": 0,
    "urinary_bladder": 1,
    "bone_hips": 2,
    "obturator_internus": 3,
    "transition_zone_prostate": 4,
    "central_zone_prostate": 5,
    "rectum": 6,
    "seminal_vesicles": 7,
    "neurovascular_bundle": 8,
}


def resample_image(image, target_spacing, is_mask=False):
    """Resample SimpleITK image to target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    return resampler.Execute(image)


def get_network_downsampling_factor():
    """
    Returns the total downsampling factor (D, H, W) for the network.
    Images must be padded to be divisible by these factors.
    """
    return (4, 16, 16)


def pad_to_divisible(image, mask, divisible_by=(4, 16, 16)):
    """
    Pad image and mask so dimensions are divisible by the given factors.

    Args:
        image: tensor of shape (C, D, H, W)
        mask: tensor of shape (D, H, W)
        divisible_by: tuple of (D_factor, H_factor, W_factor)

    Returns:
        padded_image, padded_mask, original_shape
    """
    d, h, w = image.shape[1], image.shape[2], image.shape[3]
    d_factor, h_factor, w_factor = divisible_by

    # Calculate padding needed
    pad_d = (d_factor - d % d_factor) % d_factor
    pad_h = (h_factor - h % h_factor) % h_factor
    pad_w = (w_factor - w % w_factor) % w_factor

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # F.pad expects (W_left, W_right, H_left, H_right, D_left, D_right)
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        mask = F.pad(mask, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)

    return image, mask, (d, h, w)


def collate_fn_pad(batch):
    """
    Custom collate function to handle variable-sized images by padding to max size in batch.
    Also handles the original_shape tuple returned by the dataset.
    """
    images, masks, original_shapes = zip(*batch)

    # Find max dimensions in batch
    max_d = max(img.shape[1] for img in images)
    max_h = max(img.shape[2] for img in images)
    max_w = max(img.shape[3] for img in images)

    # Pad all images and masks to max size
    padded_images = []
    padded_masks = []

    for img, mask in zip(images, masks):
        d, h, w = img.shape[1], img.shape[2], img.shape[3]
        pad_d = max_d - d
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad: F.pad expects (W_left, W_right, H_left, H_right, D_left, D_right)
        img_padded = F.pad(img, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        mask_padded = F.pad(mask, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)

        padded_images.append(img_padded)
        padded_masks.append(mask_padded)

    return torch.stack(padded_images), torch.stack(padded_masks), original_shapes


def collate_fn_single(batch):
    """
    Simple collate function for batch_size=1 that handles the original_shape.
    """
    images, masks, original_shapes = zip(*batch)
    return torch.stack(images), torch.stack(masks), original_shapes


class ElasticDeformation3D:
    """
    3D Elastic/B-spline Deformation for medical image augmentation.

    This creates smooth, anatomically plausible deformation fields using cubic B-splines
    that warp both the image and segmentation mask together. This simulates natural
    anatomical variation between patients (e.g., prostate size, shape differences).
    """

    def __init__(self,
                 alpha_range=(0, 1000),
                 sigma_range=(9, 13),
                 p=0.5):
        """
        Args:
            alpha_range: Range for random alpha (deformation strength).
                        Higher values = stronger deformations.
                        Typical: (0, 1000) for medical images
            sigma_range: Range for Gaussian smoothing sigma (deformation smoothness).
                        Higher values = smoother, more global deformations.
                        Typical: (9, 13) for B-spline-like smoothness
            p: Probability of applying the deformation (0.0 to 1.0)
        """
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, mask):
        """
        Apply elastic deformation to 3D image and mask.

        Args:
            image: numpy array of shape (D, H, W)
            mask: numpy array of shape (D, H, W)

        Returns:
            Deformed image and mask as numpy arrays
        """
        if np.random.random() > self.p:
            return image, mask

        shape = image.shape

        # Random deformation parameters
        alpha = np.random.uniform(*self.alpha_range)
        sigma = np.random.uniform(*self.sigma_range)

        # Generate random displacement fields for each dimension
        # These are like control points for a B-spline
        dx = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        # Create coordinate grid
        d, h, w = shape
        d_coords, h_coords, w_coords = np.meshgrid(
            np.arange(d), np.arange(h), np.arange(w), indexing='ij'
        )

        # Apply deformation to coordinates
        indices = np.stack([
            d_coords + dx,
            h_coords + dy,
            w_coords + dz
        ])

        # Interpolate image (linear for smooth appearance)
        deformed_image = map_coordinates(image, indices, order=1, mode='reflect')

        # Interpolate mask (nearest neighbor to preserve label integrity)
        deformed_mask = map_coordinates(mask, indices, order=0, mode='reflect')

        return deformed_image, deformed_mask


class MedicalImageDataset(Dataset):
    """
    Dataset for loading 3D medical images and segmentation masks with augmentation.
    """

    def __init__(self,
                 image_files,
                 mask_files,
                 target_spacing=TARGET_SPACING,
                 augment=False,
                 elastic_params=None):
        """
        Args:
            image_files: List of paths to image files
            mask_files: List of paths to corresponding mask files
            target_spacing: Target spacing (x, y, z) in mm for resampling
            augment: If True, apply elastic deformations
            elastic_params: Dict with elastic deformation parameters:
                - alpha_range: tuple (min, max) for deformation strength
                - sigma_range: tuple (min, max) for deformation smoothness
                - p: probability of applying deformation
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.target_spacing = target_spacing
        self.divisible_by = get_network_downsampling_factor()
        self.augment = augment

        assert len(image_files) == len(mask_files), "Mismatch between images and masks"

        # Initialize elastic deformation
        if self.augment:
            if elastic_params is None:
                elastic_params = {
                    'alpha_range': (0, 1000),
                    'sigma_range': (9, 13),
                    'p': 0.5
                }
            self.elastic_transform = ElasticDeformation3D(**elastic_params)
        else:
            self.elastic_transform = None

    def __len__(self):
        return len(self.image_files)

    def normalize_image(self, image):
        """Z-score normalization"""
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        return image

    def __getitem__(self, idx):
        # Load image and mask
        image_sitk = sitk.ReadImage(str(self.image_files[idx]))
        mask_sitk = sitk.ReadImage(str(self.mask_files[idx]))

        # Resample to target spacing
        image_sitk = resample_image(image_sitk, self.target_spacing, is_mask=False)
        mask_sitk = resample_image(mask_sitk, self.target_spacing, is_mask=True)

        image = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        mask = sitk.GetArrayFromImage(mask_sitk).astype(np.int64)

        # Store original shape before any modifications
        original_shape = image.shape  # (D, H, W)

        # Apply elastic deformation before normalization (if augmentation enabled)
        if self.augment and self.elastic_transform is not None:
            image, mask = self.elastic_transform(image, mask)

        # Normalize image
        image = self.normalize_image(image)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)
        mask_tensor = torch.from_numpy(mask).long()  # (D, H, W)

        # Pad to be divisible by network downsampling factor
        image_tensor, mask_tensor, _ = pad_to_divisible(
            image_tensor, mask_tensor, self.divisible_by
        )

        return image_tensor, mask_tensor, original_shape


def visualize_augmentation_3d(dataset_augmented, dataset_original, idx=0, num_slices=5):
    """
    Visualize the effect of 3D elastic deformation on a sample.

    Args:
        dataset_augmented: Dataset with augmentation enabled
        dataset_original: Dataset without augmentation
        idx: Index of the sample to visualize
        num_slices: Number of axial slices to display
    """
    # Get original and augmented samples
    img_aug, mask_aug, _ = dataset_augmented[idx]
    img_orig, mask_orig, _ = dataset_original[idx]

    # Convert to numpy for visualization
    img_aug = img_aug.squeeze(0).numpy()  # Remove channel dimension
    mask_aug = mask_aug.numpy()
    img_orig = img_orig.squeeze(0).numpy()
    mask_orig = mask_orig.numpy()

    # Select slices evenly distributed through the volume
    depth = img_orig.shape[0]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    # Create colormap for segmentation masks (9 classes)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors[0] = [0, 0, 0, 1]  # Background = black
    cmap = ListedColormap(colors)

    # Create figure with subplots
    fig, axes = plt.subplots(4, num_slices, figsize=(4*num_slices, 16))

    for i, slice_idx in enumerate(slice_indices):
        # Original image
        axes[0, i].imshow(img_orig[slice_idx], cmap='gray')
        axes[0, i].set_title(f'Original Image\nSlice {slice_idx}/{depth}')
        axes[0, i].axis('off')

        # Original mask
        axes[1, i].imshow(mask_orig[slice_idx], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[1, i].set_title(f'Original Mask')
        axes[1, i].axis('off')

        # Augmented image
        axes[2, i].imshow(img_aug[slice_idx], cmap='gray')
        axes[2, i].set_title(f'Augmented Image')
        axes[2, i].axis('off')

        # Augmented mask
        axes[3, i].imshow(mask_aug[slice_idx], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[3, i].set_title(f'Augmented Mask')
        axes[3, i].axis('off')

    # Add colorbar for mask labels
    label_names = list(LABELS.keys())
    cbar = plt.colorbar(axes[1, -1].images[0], ax=axes[1, -1], fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(label_names)))
    cbar.set_ticklabels(label_names, fontsize=8)

    plt.tight_layout()
    plt.suptitle('3D Elastic/B-spline Deformation Augmentation', fontsize=16, y=1.001)

    # Save figure
    save_path = Path('augmentation_visualization_3d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    plt.show()

    return fig


def visualize_multiple_augmentations(dataset_augmented, dataset_original, idx=0, n_augmentations=5, slice_idx=None):
    """
    Visualize multiple different augmentation samples from the same original image.
    This demonstrates the random variation in elastic deformations.

    Args:
        dataset_augmented: Dataset with augmentation enabled
        dataset_original: Dataset without augmentation
        idx: Index of the sample to visualize
        n_augmentations: Number of different augmentations to generate
        slice_idx: Specific slice to visualize (if None, uses middle slice)
    """
    # Get original sample
    img_orig, mask_orig, _ = dataset_original[idx]
    img_orig = img_orig.squeeze(0).numpy()
    mask_orig = mask_orig.numpy()

    # Select slice
    if slice_idx is None:
        slice_idx = img_orig.shape[0] // 2

    # Create colormap
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors[0] = [0, 0, 0, 1]
    cmap = ListedColormap(colors)

    # Create figure
    fig, axes = plt.subplots(2, n_augmentations + 1, figsize=(4*(n_augmentations + 1), 8))

    # Display original
    axes[0, 0].imshow(img_orig[slice_idx], cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(mask_orig[slice_idx], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')

    # Generate and display multiple augmentations
    for i in range(n_augmentations):
        img_aug, mask_aug, _ = dataset_augmented[idx]
        img_aug = img_aug.squeeze(0).numpy()
        mask_aug = mask_aug.numpy()

        axes[0, i+1].imshow(img_aug[slice_idx], cmap='gray')
        axes[0, i+1].set_title(f'Augmentation {i+1}')
        axes[0, i+1].axis('off')

        axes[1, i+1].imshow(mask_aug[slice_idx], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[1, i+1].set_title(f'Augmentation {i+1}')
        axes[1, i+1].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Multiple Random Elastic Deformations (Slice {slice_idx})', fontsize=14, y=1.001)

    # Save figure
    save_path = Path('augmentation_variations_3d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Variation visualization saved to: {save_path}")

    plt.show()

    return fig


# Demo/testing code
if __name__ == "__main__":
    """
    Demo script to visualize 3D elastic deformations.
    Run this file directly to see the augmentation in action.
    """


    # Configuration
    data_dir = Path(r"./7013610/data/data")

    # Find image files
    image_files = sorted(data_dir.glob("*_img.nii*"))

    # Filter out prediction files and prepare lists
    all_images = []
    all_masks = []

    for img_file in image_files:
        # Skip prediction files
        if 'pred' in img_file.name:
            continue

        # Construct corresponding mask file
        mask_file = img_file.parent / img_file.name.replace('_img', '_mask')

        if not mask_file.exists():
            continue

        all_images.append(img_file)
        all_masks.append(mask_file)

    print(f"\nFound {len(all_images)} image-mask pairs")

    if len(all_images) == 0:
        print("Error: No data found!")
        print(f"Please check that data exists in: {data_dir}")
        exit(1)


    # Dataset without augmentation
    dataset_original = MedicalImageDataset(
        all_images[:5],  # Use first 5 samples for demo
        all_masks[:5],
        augment=False
    )

    # Dataset with elastic deformation
    elastic_params = {
        'alpha_range': (0, 1000),    # Deformation strength
        'sigma_range': (9, 13),       # Smoothness (B-spline-like)
        'p': 1.0                      # Always apply for visualization
    }

    dataset_augmented = MedicalImageDataset(
        all_images[:5],
        all_masks[:5],
        augment=True,
        elastic_params=elastic_params
    )

    print("\nAugmentation Parameters:")
    print(f"  - Alpha range (deformation strength): {elastic_params['alpha_range']}")
    print(f"  - Sigma range (smoothness): {elastic_params['sigma_range']}")
    print(f"  - Probability: {elastic_params['p']}")

    # Visualize first sample
    print("\n" + "=" * 80)
    print("Generating Visualizations...")
    print("=" * 80)

    print("\n1. Comparing original vs augmented across multiple slices...")
    visualize_augmentation_3d(
        dataset_augmented,
        dataset_original,
        idx=0,
        num_slices=5
    )

    print("\n2. Showing multiple random augmentations of the same slice...")
    visualize_multiple_augmentations(
        dataset_augmented,
        dataset_original,
        idx=0,
        n_augmentations=5
    )


