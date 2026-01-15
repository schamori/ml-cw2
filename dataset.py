"""
3D Medical Image Dataset
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import SimpleITK as sitk

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
            augment: Unused (kept for backward compatibility)
            elastic_params: Unused (kept for backward compatibility)
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.target_spacing = target_spacing
        self.divisible_by = get_network_downsampling_factor()

        assert len(image_files) == len(mask_files), "Mismatch between images and masks"

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




