import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import SimpleITK as sitk
from datetime import datetime
import json
from tqdm import tqdm

# Import nnUNet components
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op

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

# Foreground labels only (excluding background)
FOREGROUND_LABELS = {k: v for k, v in LABELS.items() if k != "background"}

# Target spacing for resampling (median of dataset)
TARGET_SPACING = (1.0, 1.0, 2.0)


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


class DiceCELoss(nn.Module):
    """Combined Dice Loss + Cross Entropy Loss"""

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):
        """
        Calculate Dice loss for all classes
        pred: (B, C, D, H, W) - raw logits
        target: (B, D, H, W) - class indices
        """
        # Convert logits to probabilities
        pred_softmax = F.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        # Calculate dice score per class
        dice_scores = []
        for class_idx in range(self.num_classes):
            pred_class = pred_softmax[:, class_idx]
            target_class = target_one_hot[:, class_idx]

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()

            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)

        # Average dice loss (1 - dice score)
        dice_loss = 1.0 - torch.stack(dice_scores).mean()

        return dice_loss

    def forward(self, pred, target):
        """
        pred: (B, C, D, H, W)
        target: (B, D, H, W)
        """
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss, ce_loss, dice_loss


def calculate_dice_per_class(pred, target, num_classes, smooth=1e-5, include_background=False):
    """
    Calculate Dice score for each class (excluding background by default)

    pred: (B, D, H, W) - predicted class indices
    target: (B, D, H, W) - ground truth class indices
    include_background: if False, excludes class 0 (background) from calculation
    """
    dice_scores = {}
    label_names = {int(v): k for k, v in LABELS.items()}

    start_class = 0 if include_background else 1  # Skip background if not included

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


class MedicalImageDataset(Dataset):
    """Dataset for loading whole medical images and segmentation masks"""

    def __init__(self, image_files, mask_files, target_spacing=TARGET_SPACING):
        """
        Args:
            image_files: List of paths to image files
            mask_files: List of paths to corresponding mask files
            target_spacing: Target spacing (x, y, z) in mm for resampling
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


def unpad_output(output, original_shape):
    """
    Remove padding from network output to match original image shape.

    Args:
        output: tensor of shape (B, C, D, H, W) or (B, D, H, W)
        original_shape: tuple of (D, H, W)

    Returns:
        Unpadded output
    """
    d, h, w = original_shape
    if output.dim() == 5:  # (B, C, D, H, W)
        return output[:, :, :d, :h, :w]
    else:  # (B, D, H, W)
        return output[:, :d, :h, :w]


def build_nnunet_network(num_input_channels=1, num_classes=9):
    """
    Build a nnUNet-style 3D U-Net adapted for prostate MRI dimensions.

    Typical image sizes: D=25-40, H=180-320, W=180-320

    Total downsampling: D=4x, H=16x, W=16x
    Images should be padded to be divisible by (4, 16, 16)
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

    # Strides: be conservative with depth (D) downsampling since D is small (25-40)
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
    
    # Decoder remains plain in standard nnU-Net ResNet implementation
    n_conv_per_stage_decoder = [2, 2, 2, 2]

    num_stages = len(kernel_sizes)

    # Feature channels per stage
    base_features = 32
    max_features = 320
    features_per_stage = [min(base_features * 2 ** i, max_features) for i in range(num_stages)]
    # Results in: [32, 64, 128, 256, 320]

    network = ResidualEncoderUNet(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
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


class Trainer:
    """Training manager for nnUNet"""

    def __init__(self, model, train_loader, val_loader, test_loader, device, num_classes,
                 learning_rate=1e-3, log_dir='logs', checkpoint_dir='checkpoints'):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

        self.criterion = DiceCELoss(num_classes=num_classes, ce_weight=1.0, dice_weight=1.0)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # Logging setup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.json"

        self.training_log = {
            'training_started': timestamp,
            'num_classes': num_classes,
            'learning_rate': learning_rate,
            'epochs': [],
            'test_results': None
        }

        self.best_val_loss = float('inf')

    def log_message(self, message):
        """Print and log message"""
        print(message)

    def save_log(self):
        """Save training log to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (images, masks, original_shapes) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Handle deep supervision (if enabled)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Calculate loss (on padded images - padding is zeros which is background)
            total_loss, ce_loss, dice_loss = self.criterion(outputs, masks)

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Update running losses
            running_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_dice_loss += dice_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'ce': ce_loss.item(),
                'dice': dice_loss.item()
            })

        avg_loss = running_loss / len(self.train_loader)
        avg_ce_loss = running_ce_loss / len(self.train_loader)
        avg_dice_loss = running_dice_loss / len(self.train_loader)

        return avg_loss, avg_ce_loss, avg_dice_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        all_dice_scores = {k: [] for k in FOREGROUND_LABELS.keys()}

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        with torch.no_grad():
            for batch_idx, (images, masks, original_shapes) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Handle deep supervision
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                # Calculate loss (on padded data for consistency with training)
                total_loss, ce_loss, dice_loss = self.criterion(outputs, masks)

                # Update running losses
                running_loss += total_loss.item()
                running_ce_loss += ce_loss.item()
                running_dice_loss += dice_loss.item()

                # Calculate Dice scores per class (excluding background)
                # Unpad predictions and masks to original size for accurate metrics
                predictions = torch.argmax(outputs, dim=1)

                for i in range(predictions.shape[0]):
                    orig_shape = original_shapes[i]
                    pred_unpadded = predictions[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
                    mask_unpadded = masks[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]

                    dice_scores = calculate_dice_per_class(
                        pred_unpadded.unsqueeze(0),
                        mask_unpadded.unsqueeze(0),
                        self.num_classes,
                        include_background=False
                    )

                    for class_name, score in dice_scores.items():
                        if not np.isnan(score):
                            all_dice_scores[class_name].append(score)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'ce': ce_loss.item(),
                    'dice': dice_loss.item()
                })

        avg_loss = running_loss / len(self.val_loader)
        avg_ce_loss = running_ce_loss / len(self.val_loader)
        avg_dice_loss = running_dice_loss / len(self.val_loader)

        # Average Dice scores per class
        mean_dice_scores = {}
        for class_name, scores in all_dice_scores.items():
            if len(scores) > 0:
                mean_dice_scores[class_name] = np.mean(scores)
            else:
                mean_dice_scores[class_name] = float('nan')

        return avg_loss, avg_ce_loss, avg_dice_loss, mean_dice_scores

    def test(self):
        """Evaluate on test set"""
        self.log_message("\n" + "=" * 80)
        self.log_message("Evaluating on Test Set")
        self.log_message("=" * 80)

        self.model.eval()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        all_dice_scores = {k: [] for k in FOREGROUND_LABELS.keys()}

        progress_bar = tqdm(self.test_loader, desc="Test")

        with torch.no_grad():
            for batch_idx, (images, masks, original_shapes) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Handle deep supervision
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                # Calculate loss
                total_loss, ce_loss, dice_loss = self.criterion(outputs, masks)

                # Update running losses
                running_loss += total_loss.item()
                running_ce_loss += ce_loss.item()
                running_dice_loss += dice_loss.item()

                # Calculate Dice scores per class (excluding background)
                # Unpad predictions and masks to original size for accurate metrics
                predictions = torch.argmax(outputs, dim=1)

                for i in range(predictions.shape[0]):
                    orig_shape = original_shapes[i]
                    pred_unpadded = predictions[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
                    mask_unpadded = masks[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]

                    dice_scores = calculate_dice_per_class(
                        pred_unpadded.unsqueeze(0),
                        mask_unpadded.unsqueeze(0),
                        self.num_classes,
                        include_background=False
                    )

                    for class_name, score in dice_scores.items():
                        if not np.isnan(score):
                            all_dice_scores[class_name].append(score)

                progress_bar.set_postfix({
                    'loss': total_loss.item()
                })

        avg_loss = running_loss / len(self.test_loader)
        avg_ce_loss = running_ce_loss / len(self.test_loader)
        avg_dice_loss = running_dice_loss / len(self.test_loader)

        # Average Dice scores per class
        mean_dice_scores = {}
        for class_name, scores in all_dice_scores.items():
            if len(scores) > 0:
                mean_dice_scores[class_name] = np.mean(scores)
            else:
                mean_dice_scores[class_name] = float('nan')

        # Calculate mean Dice across all foreground classes
        valid_scores = [s for s in mean_dice_scores.values() if not np.isnan(s)]
        mean_foreground_dice = np.mean(valid_scores) if valid_scores else float('nan')

        # Log results
        self.log_message(f"\nTest Results:")
        self.log_message(f"  Total Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, Dice: {avg_dice_loss:.4f})")
        self.log_message(f"\n  Dice Scores per Class (excluding background):")
        for class_name, score in mean_dice_scores.items():
            if not np.isnan(score):
                self.log_message(f"    {class_name:30s}: {score:.4f}")
            else:
                self.log_message(f"    {class_name:30s}: N/A (not present)")
        self.log_message(f"\n  Mean Foreground Dice: {mean_foreground_dice:.4f}")

        # Save test results
        test_results = {
            'total_loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'dice_loss': avg_dice_loss,
            'dice_scores_per_class': mean_dice_scores,
            'mean_foreground_dice': mean_foreground_dice
        }
        self.training_log['test_results'] = test_results
        self.save_log()

        return test_results

    def train(self, num_epochs):
        """Main training loop"""
        self.log_message("=" * 80)
        self.log_message("Starting nnUNet Training")
        self.log_message("=" * 80)
        self.log_message(f"Device: {self.device}")
        self.log_message(f"Number of classes: {self.num_classes}")
        self.log_message(f"Number of epochs: {num_epochs}")
        self.log_message(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.log_message(f"Training samples: {len(self.train_loader.dataset)}")
        self.log_message(f"Validation samples: {len(self.val_loader.dataset)}")
        self.log_message(f"Test samples: {len(self.test_loader.dataset)}")
        self.log_message("=" * 80)

        for epoch in range(1, num_epochs + 1):
            self.log_message(f"\nEpoch {epoch}/{num_epochs}")
            self.log_message("-" * 80)

            # Training
            train_loss, train_ce, train_dice = self.train_epoch(epoch)

            # Validation
            val_loss, val_ce, val_dice, dice_scores = self.validate_epoch(epoch)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Calculate mean foreground Dice
            valid_scores = [s for s in dice_scores.values() if not np.isnan(s)]
            mean_foreground_dice = np.mean(valid_scores) if valid_scores else float('nan')

            # Log results
            epoch_log = {
                'epoch': epoch,
                'train': {
                    'total_loss': train_loss,
                    'ce_loss': train_ce,
                    'dice_loss': train_dice
                },
                'validation': {
                    'total_loss': val_loss,
                    'ce_loss': val_ce,
                    'dice_loss': val_dice,
                    'dice_scores_per_class': dice_scores,
                    'mean_foreground_dice': mean_foreground_dice
                },
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_log['epochs'].append(epoch_log)
            self.save_log()

            # Print epoch summary
            self.log_message(f"\nEpoch {epoch} Summary:")
            self.log_message(f"  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Dice: {train_dice:.4f})")
            self.log_message(f"  Val Loss: {val_loss:.4f} (CE: {val_ce:.4f}, Dice: {val_dice:.4f})")
            self.log_message(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.log_message(f"\n  Dice Scores per Class (excluding background):")
            for class_name, score in dice_scores.items():
                if not np.isnan(score):
                    self.log_message(f"    {class_name:30s}: {score:.4f}")
                else:
                    self.log_message(f"    {class_name:30s}: N/A (not present)")
            self.log_message(f"\n  Mean Foreground Dice: {mean_foreground_dice:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'dice_scores': dice_scores,
                    'mean_foreground_dice': mean_foreground_dice
                }, checkpoint_path)
                self.log_message(f"\n  *** New best model saved! Val Loss: {val_loss:.4f} ***")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                self.log_message(f"  Checkpoint saved: {checkpoint_path}")

        self.log_message("\n" + "=" * 80)
        self.log_message("Training completed!")
        self.log_message(f"Best validation loss: {self.best_val_loss:.4f}")
        self.log_message("=" * 80)

        # Load best model and evaluate on test set
        best_checkpoint = self.checkpoint_dir / 'best_model.pth'
        if best_checkpoint.exists():
            self.log_message("\nLoading best model for test evaluation...")
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        test_results = self.test()

        self.log_message(f"\nTraining log saved to: {self.log_file}")
        self.log_message("=" * 80)

        return test_results


def prepare_data_lists(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare lists of training, validation, and test files

    Args:
        data_dir: Directory containing the data
        train_ratio: Proportion of data for training (default 0.7)
        val_ratio: Proportion of data for validation (default 0.15)
        test_ratio: Proportion of data for testing (default 0.15)

    Returns:
        Tuple of (train_images, train_masks, val_images, val_masks, test_images, test_masks)
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


def main():
    """Main training function"""

    # Configuration
    data_dir = Path(r"./7013610/data/data")
    batch_size = 1  # Use batch size 1 for whole images (variable sizes)
    num_epochs = 100
    learning_rate = 1e-3
    num_workers = 0  # Set to 0 on Windows to avoid multiprocessing issues

    # Data split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    print("\nPreparing data...")
    train_images, train_masks, val_images, val_masks, test_images, test_masks = prepare_data_lists(
        data_dir, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )

    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")

    if len(train_images) == 0 or len(val_images) == 0 or len(test_images) == 0:
        print("Error: Not enough data for training/validation/testing!")
        return

    # Create datasets (whole images, no patches)
    train_dataset = MedicalImageDataset(train_images, train_masks)
    val_dataset = MedicalImageDataset(val_images, val_masks)
    test_dataset = MedicalImageDataset(test_images, test_masks)

    # Create dataloaders with custom collate function for variable sizes
    # Use collate_fn_single for batch_size=1, collate_fn_pad for larger batches
    collate_fn = collate_fn_single if batch_size == 1 else collate_fn_pad

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )

    # Build model
    print("\nBuilding model...")
    num_classes = len(LABELS)
    model = build_nnunet_network(num_input_channels=1, num_classes=num_classes)
    model = model.to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        learning_rate=learning_rate
    )

    # Start training
    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()