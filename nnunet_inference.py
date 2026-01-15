import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from pathlib import Path
import SimpleITK as sitk

# Import nnUNet components
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

# Target spacing (must match training)
TARGET_SPACING = (1.0, 1.0, 2.0)

# Network downsampling factor (must match training)
NETWORK_DIVISIBLE_BY = (4, 16, 16)  # (D, H, W)


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


def resample_to_original(prediction_sitk, reference_sitk):
    """Resample prediction back to original image spacing/size."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_sitk.GetSpacing())
    resampler.SetSize(reference_sitk.GetSize())
    resampler.SetOutputDirection(reference_sitk.GetDirection())
    resampler.SetOutputOrigin(reference_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Always nearest for masks

    return resampler.Execute(prediction_sitk)


def build_nnunet_network(num_input_channels=1, num_classes=9):
    """
    Build a nnUNet-style 3D U-Net adapted for prostate MRI dimensions.
    Must match the training configuration exactly.

    Total downsampling: D=4x, H=16x, W=16x
    """
    conv_op = torch.nn.Conv3d

    # 5-stage network with reduced depth downsampling
    kernel_sizes = [
        [3, 3, 3],  # stage 0
        [3, 3, 3],  # stage 1
        [3, 3, 3],  # stage 2
        [3, 3, 3],  # stage 3
        [3, 3, 3],  # stage 4
    ]

    # Strides: conservative with depth (D) downsampling
    # Total D downsampling: 1*1*2*2*1 = 4
    # Total H/W downsampling: 1*2*2*2*2 = 16
    strides = [
        [1, 1, 1],  # stage 0: no downsampling
        [1, 2, 2],  # stage 1: downsample H,W only
        [2, 2, 2],  # stage 2: downsample all
        [2, 2, 2],  # stage 3: downsample all
        [1, 2, 2],  # stage 4: downsample H,W only
    ]

    # Residual architectures usually use 2 blocks per stage
    # In ResidualEncoderUNet, these are blocks of (conv-norm-relu-conv-norm-relu) 
    # with a skip connection.
    n_blocks_per_stage = [2, 2, 2, 2, 2]

    n_conv_per_stage_decoder = [2, 2, 2, 2]
    num_stages = len(kernel_sizes)

    # Feature maps per stage: [32, 64, 128, 256, 320]
    base_features = 32
    max_features = 320
    features_per_stage = [min(base_features * 2 ** i, max_features) for i in range(num_stages)]

    network = ResidualEncoderUNet(
        input_channels=num_input_channels,
        n_stages=num_stages,
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

    return network


def pad_to_divisible(image, divisible_by):
    """
    Pad image so dimensions are divisible by the given factors.

    Args:
        image: numpy array of shape (D, H, W)
        divisible_by: tuple of (D_factor, H_factor, W_factor)

    Returns:
        padded_image, padding tuple for unpadding later
    """
    d, h, w = image.shape
    d_factor, h_factor, w_factor = divisible_by

    pad_d = (d_factor - d % d_factor) % d_factor
    pad_h = (h_factor - h % h_factor) % h_factor
    pad_w = (w_factor - w % w_factor) % w_factor

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    return image, (pad_d, pad_h, pad_w)


def unpad(image, padding, original_shape):
    """Remove padding from image."""
    d, h, w = original_shape
    return image[:d, :h, :w]


def run_inference(image_path, output_path, checkpoint_path=None):
    """
    Run nnUNet inference.

    Args:
        image_path: Path to input image
        output_path: Path to save prediction
        checkpoint_path: Path to model checkpoint (if None, uses random weights)
    """
    print("=" * 60)
    print("nnUNet Inference")
    print("=" * 60)

    # Load image
    print(f"\nLoading image: {image_path}")
    img_sitk_original = sitk.ReadImage(str(image_path))

    print(f"Original image size: {img_sitk_original.GetSize()}")
    print(f"Original spacing: {img_sitk_original.GetSpacing()}")

    # Resample to target spacing
    print(f"\nResampling to target spacing: {TARGET_SPACING}")
    img_sitk_resampled = resample_image(img_sitk_original, TARGET_SPACING, is_mask=False)

    print(f"Resampled image size: {img_sitk_resampled.GetSize()}")

    # Convert to numpy
    image = sitk.GetArrayFromImage(img_sitk_resampled).astype(np.float32)  # Shape: (D, H, W)
    print(f"Array shape (D, H, W): {image.shape}")

    # Normalize (z-score)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std

    # Store shape before padding
    resampled_shape = image.shape

    # Pad to be divisible by network downsampling factor
    image, padding = pad_to_divisible(image, NETWORK_DIVISIBLE_BY)
    print(f"Padded shape: {image.shape} (padding: D+{padding[0]}, H+{padding[1]}, W+{padding[2]})")

    # Prepare tensor: (1, 1, D, H, W) for batch and channel
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    print(f"Input tensor shape: {image_tensor.shape}")

    # Build network
    num_classes = len(LABELS)
    print(f"\nBuilding nnUNet with {num_classes} output classes...")
    network = build_nnunet_network(num_input_channels=1, num_classes=num_classes)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("WARNING: Using random weights (no checkpoint provided)")

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)
    network.eval()

    print(f"Device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in network.parameters()):,}")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = network(image_tensor)

        if isinstance(output, (list, tuple)):
            output = output[0]

        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

    print(f"Prediction shape (padded): {prediction.shape}")

    # Unpad prediction
    prediction = unpad(prediction, padding, resampled_shape)
    print(f"Prediction shape (unpadded): {prediction.shape}")

    # Convert prediction to SimpleITK image (at resampled spacing)
    pred_sitk_resampled = sitk.GetImageFromArray(prediction.astype(np.uint8))
    pred_sitk_resampled.SetSpacing(TARGET_SPACING)
    pred_sitk_resampled.SetOrigin(img_sitk_resampled.GetOrigin())
    pred_sitk_resampled.SetDirection(img_sitk_resampled.GetDirection())

    # Resample prediction back to original spacing
    print(f"\nResampling prediction back to original spacing: {img_sitk_original.GetSpacing()}")
    pred_sitk_final = resample_to_original(pred_sitk_resampled, img_sitk_original)
    print(f"Final prediction size: {pred_sitk_final.GetSize()}")

    # Analyze prediction
    prediction_final = sitk.GetArrayFromImage(pred_sitk_final)
    unique, counts = np.unique(prediction_final, return_counts=True)
    print("\nPredicted class distribution:")
    label_names = {int(v): k for k, v in LABELS.items()}
    for class_id, count in zip(unique, counts):
        label_name = label_names.get(int(class_id), "unknown")
        percentage = (count / prediction_final.size) * 100
        print(f"  Class {class_id} ({label_name}): {count:,} voxels ({percentage:.2f}%)")

    # Save prediction
    print(f"\nSaving prediction to: {output_path}")
    sitk.WriteImage(pred_sitk_final, str(output_path))

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)

    return pred_sitk_final


if __name__ == "__main__":
    # Paths
    data_dir = Path(r"./7013610/data/data")
    source_image = data_dir / "001000_img.nii"
    output_file = data_dir / "001000_pred_nnunet.nii.gz"

    # Set to your checkpoint path, or None for random weights
    checkpoint_path = "checkpoints/checkpoint_epoch_40.pth"

    run_inference(source_image, output_file, checkpoint_path=checkpoint_path)

    print(f"\nFinal output: {output_file}")