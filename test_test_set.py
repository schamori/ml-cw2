import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm

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

FOREGROUND_LABELS = {k: v for k, v in LABELS.items() if k != "background"}

TARGET_SPACING = (1.0, 1.0, 2.0)
NETWORK_DIVISIBLE_BY = (4, 16, 16)


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


def build_nnunet_network(num_input_channels=1, num_classes=9):
    """Build nnUNet matching training config."""
    conv_op = torch.nn.Conv3d

    kernel_sizes = [[3, 3, 3]] * 5
    strides = [
        [1, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2],
    ]

    # Residual architectures usually use 2 blocks per stage
    # In ResidualEncoderUNet, these are blocks of (conv-norm-relu-conv-norm-relu) 
    # with a skip connection.
    n_blocks_per_stage = [2, 2, 2, 2, 2] 
    n_conv_per_stage_decoder = [2, 2, 2, 2]

    base_features = 32
    max_features = 320
    features_per_stage = [min(base_features * 2 ** i, max_features) for i in range(5)]

    return ResidualEncoderUNet(
        input_channels=num_input_channels,
        n_stages=5,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_blocks_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=get_matching_instancenorm(conv_op),
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False
    )


def pad_to_divisible(image, divisible_by):
    """Pad image so dimensions are divisible by the given factors."""
    d, h, w = image.shape
    d_factor, h_factor, w_factor = divisible_by

    pad_d = (d_factor - d % d_factor) % d_factor
    pad_h = (h_factor - h % h_factor) % h_factor
    pad_w = (w_factor - w % w_factor) % w_factor

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    return image, (pad_d, pad_h, pad_w)


def prepare_data_lists(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Same split function as training - uses seed 42 for reproducibility."""
    data_dir = Path(data_dir)
    image_files = sorted(data_dir.glob("*_img.nii*"))

    all_images = []
    all_masks = []

    for img_file in image_files:
        mask_file = img_file.parent / img_file.name.replace('_img', '_mask')
        if mask_file.exists():
            all_images.append(img_file)
            all_masks.append(mask_file)

    # Same seed as training!
    np.random.seed(42)
    indices = np.random.permutation(len(all_images))

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    test_indices = indices[n_train + n_val:]

    test_images = [all_images[i] for i in test_indices]
    test_masks = [all_masks[i] for i in test_indices]

    return test_images, test_masks


def calculate_dice(pred, target, num_classes, smooth=1e-5):
    """Calculate Dice score for each foreground class."""
    dice_scores = {}
    label_names = {int(v): k for k, v in LABELS.items()}

    for class_idx in range(1, num_classes):  # Skip background
        pred_class = (pred == class_idx).astype(np.float32)
        target_class = (target == class_idx).astype(np.float32)

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        if union > 0:
            dice = (2.0 * intersection + smooth) / (union + smooth)
        else:
            dice = float('nan')

        class_name = label_names.get(class_idx, f"class_{class_idx}")
        dice_scores[class_name] = dice

    return dice_scores


def evaluate_test_set(data_dir, checkpoint_path):
    """Evaluate model on test set and print dice scores."""
    print("=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)

    # Get test set (same split as training)
    test_images, test_masks = prepare_data_lists(data_dir)
    print(f"\nTest set size: {len(test_images)} images")

    # Build and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    num_classes = len(LABELS)
    model = build_nnunet_network(num_input_channels=1, num_classes=num_classes)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    model = model.to(device)
    model.eval()

    # Collect dice scores for all images
    all_dice_scores = {k: [] for k in FOREGROUND_LABELS.keys()}

    print("\nRunning inference on test set...")
    for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
        # Load and resample image
        img_sitk = sitk.ReadImage(str(img_path))
        img_sitk = resample_image(img_sitk, TARGET_SPACING, is_mask=False)
        image = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

        # Load and resample mask
        mask_sitk = sitk.ReadImage(str(mask_path))
        mask_sitk = resample_image(mask_sitk, TARGET_SPACING, is_mask=True)
        mask = sitk.GetArrayFromImage(mask_sitk).astype(np.int64)

        # Normalize
        mean, std = image.mean(), image.std()
        if std > 0:
            image = (image - mean) / std

        # Pad
        original_shape = image.shape
        image, padding = pad_to_divisible(image, NETWORK_DIVISIBLE_BY)

        # Inference
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]

        # Unpad
        pred = pred[:original_shape[0], :original_shape[1], :original_shape[2]]

        # Calculate dice
        dice_scores = calculate_dice(pred, mask, num_classes)

        for class_name, score in dice_scores.items():
            if not np.isnan(score):
                all_dice_scores[class_name].append(score)

    # Print results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"\n{'Class':<30} {'Mean Dice':>12} {'Std':>10} {'N':>6}")
    print("-" * 60)

    valid_means = []
    for class_name in FOREGROUND_LABELS.keys():
        scores = all_dice_scores[class_name]
        if len(scores) > 0:
            mean_dice = np.mean(scores)
            std_dice = np.std(scores)
            valid_means.append(mean_dice)
            print(f"{class_name:<30} {mean_dice:>12.4f} {std_dice:>10.4f} {len(scores):>6}")
        else:
            print(f"{class_name:<30} {'N/A':>12} {'N/A':>10} {0:>6}")

    print("-" * 60)
    if valid_means:
        overall_mean = np.mean(valid_means)
        print(f"{'MEAN FOREGROUND DICE':<30} {overall_mean:>12.4f}")
    print("=" * 70)

    return all_dice_scores


if __name__ == "__main__":
    data_dir = Path("./7013610/data/data")
    checkpoint_path = "checkpoints/best_model.pth"

    evaluate_test_set(data_dir, checkpoint_path)