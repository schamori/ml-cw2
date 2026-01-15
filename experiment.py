
from datetime import datetime
import json
from pathlib import Path
import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm

from dataset import (
    MedicalImageDataset,
    collate_fn_pad,
    collate_fn_single,
    LABELS
)
from losses import DiceCELoss, compute_weighted_dice_score
import weighted_matrices

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

FOREGROUND_LABELS = {k: v for k, v in LABELS.items() if k != "background"}
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_dice_per_class(pred, target, num_classes, smooth=1e-5, include_background=False):
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


def build_network(num_input_channels=1, num_classes=9, use_residual=True):
    conv_op = torch.nn.Conv3d
    kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    num_stages = len(kernel_sizes)

    base_features = 32
    max_features = 320
    features_per_stage = [min(base_features * 2 ** i, max_features) for i in range(num_stages)]

    if use_residual:
        n_blocks_per_stage = [2, 2, 2, 2, 2]
        n_conv_per_stage_decoder = [2, 2, 2, 2]
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
    else:
        n_conv_per_stage = [2, 2, 2, 2, 2]
        n_conv_per_stage_decoder = [2, 2, 2, 2]
        network = PlainConvUNet(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
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


class PolynomialLRScheduler:
    def __init__(self, optimizer, max_epochs, initial_lr, power=0.9):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.power = power
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        lr_multiplier = (1 - self.current_epoch / self.max_epochs) ** self.power
        new_lr = self.initial_lr * lr_multiplier
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, num_classes,
                 weight_matrix, learning_rate=1e-3, log_dir='logs', checkpoint_dir='checkpoints',
                 max_epochs=100, weighted=True, ce_weight=1.0, dice_weight=1.0, experiment_name=''):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.initial_lr = learning_rate
        self.experiment_name = experiment_name

        self.criterion = DiceCELoss(
            num_classes=num_classes,
            weight_matrix=weight_matrix,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            weighted=weighted
        )
        self.weight_matrix = weight_matrix
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = PolynomialLRScheduler(self.optimizer, max_epochs=max_epochs, initial_lr=learning_rate, power=0.9)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"exp_{experiment_name}_{timestamp}.json"

        self.training_log = {
            'experiment_name': experiment_name,
            'training_started': timestamp,
            'num_classes': num_classes,
            'learning_rate': learning_rate,
            'weighted': weighted,
            'ce_weight': ce_weight,
            'dice_weight': dice_weight,
            'epochs': [],
            'test_results': None
        }

        self.best_val_loss = float('inf')
        self.start_epoch = 1

    def save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0

        for images, masks, _ in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            total_loss, ce_loss, dice_loss = self.criterion(outputs, masks)
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_dice_loss += dice_loss.item()

        return running_loss / len(self.train_loader), running_ce_loss / len(self.train_loader), running_dice_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_dice_scores = {k: [] for k in FOREGROUND_LABELS.keys()}
        all_weighted_dice_scores = []

        with torch.no_grad():
            for images, masks, original_shapes in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                total_loss, _, _ = self.criterion(outputs, masks)
                running_loss += total_loss.item()

                predictions = torch.argmax(outputs, dim=1)
                for i in range(predictions.shape[0]):
                    orig_shape = original_shapes[i]
                    pred_unpadded = predictions[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
                    mask_unpadded = masks[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]

                    dice_scores = calculate_dice_per_class(pred_unpadded.unsqueeze(0), mask_unpadded.unsqueeze(0), self.num_classes, include_background=False)
                    for class_name, score in dice_scores.items():
                        if not np.isnan(score):
                            all_dice_scores[class_name].append(score)

                    weighted_scores = compute_weighted_dice_score(pred_unpadded, mask_unpadded, self.weight_matrix, num_classes=self.num_classes)
                    all_weighted_dice_scores.append(weighted_scores.cpu())

        avg_loss = running_loss / len(self.val_loader)
        mean_dice_scores = {k: np.mean(v) if v else float('nan') for k, v in all_dice_scores.items()}
        mean_weighted_dice = torch.stack(all_weighted_dice_scores).mean(dim=0) if all_weighted_dice_scores else torch.zeros(self.num_classes)
        mean_weighted_dice_dict = {f"{list(LABELS.keys())[i]}": mean_weighted_dice[i].item() for i in range(self.num_classes)}

        return avg_loss, mean_dice_scores, mean_weighted_dice_dict

    def test(self):
        print(f"\n{'=' * 60}\nEvaluating on Test Set\n{'=' * 60}")
        self.model.eval()
        all_dice_scores = {k: [] for k in FOREGROUND_LABELS.keys()}
        all_weighted_dice_scores = []

        with torch.no_grad():
            for images, masks, original_shapes in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                predictions = torch.argmax(outputs, dim=1)
                for i in range(predictions.shape[0]):
                    orig_shape = original_shapes[i]
                    pred_unpadded = predictions[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
                    mask_unpadded = masks[i, :orig_shape[0], :orig_shape[1], :orig_shape[2]]

                    dice_scores = calculate_dice_per_class(pred_unpadded.unsqueeze(0), mask_unpadded.unsqueeze(0), self.num_classes, include_background=False)
                    for class_name, score in dice_scores.items():
                        if not np.isnan(score):
                            all_dice_scores[class_name].append(score)

                    weighted_scores = compute_weighted_dice_score(pred_unpadded, mask_unpadded, self.weight_matrix, num_classes=self.num_classes)
                    all_weighted_dice_scores.append(weighted_scores.cpu())

        mean_dice_scores = {k: np.mean(v) if v else float('nan') for k, v in all_dice_scores.items()}
        valid_scores = [s for s in mean_dice_scores.values() if not np.isnan(s)]
        mean_foreground_dice = np.mean(valid_scores) if valid_scores else float('nan')

        mean_weighted_dice = torch.stack(all_weighted_dice_scores).mean(dim=0) if all_weighted_dice_scores else torch.zeros(self.num_classes)
        mean_weighted_dice_dict = {f"{list(LABELS.keys())[i]}": mean_weighted_dice[i].item() for i in range(self.num_classes)}
        mean_weighted_foreground_dice = mean_weighted_dice[1:].mean().item()

        print(f"\nTest Results for {self.experiment_name}:")
        print(f"  Mean Foreground Dice: {mean_foreground_dice:.4f}")
        print(f"  Mean Weighted Foreground Dice: {mean_weighted_foreground_dice:.4f}")

        test_results = {
            'dice_scores_per_class': mean_dice_scores,
            'mean_foreground_dice': mean_foreground_dice,
            'weighted_dice_scores_per_class': mean_weighted_dice_dict,
            'mean_weighted_foreground_dice': mean_weighted_foreground_dice
        }
        self.training_log['test_results'] = test_results
        self.save_log()
        return test_results

    def train(self, num_epochs):
        print(f"\n{'=' * 60}\nStarting Experiment: {self.experiment_name}\n{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_loss, _, _ = self.train_epoch(epoch)
            val_loss, dice_scores, weighted_dice_scores = self.validate_epoch(epoch)
            self.scheduler.step(epoch)

            valid_scores = [s for s in dice_scores.values() if not np.isnan(s)]
            mean_foreground_dice = np.mean(valid_scores) if valid_scores else float('nan')
            weighted_fg_scores = [weighted_dice_scores[k] for k in list(LABELS.keys())[1:]]
            mean_weighted_fg_dice = np.mean(weighted_fg_scores)

            epoch_log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mean_foreground_dice': mean_foreground_dice,
                'mean_weighted_foreground_dice': mean_weighted_fg_dice,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_log['epochs'].append(epoch_log)
            self.save_log()

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, FG Dice={mean_foreground_dice:.4f}, Weighted FG Dice={mean_weighted_fg_dice:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir / f'best_{self.experiment_name}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"  *** New best model saved! ***")

        best_checkpoint = self.checkpoint_dir / f'best_{self.experiment_name}.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        return self.test()


def prepare_data_lists(data_dir, train_ratio=0.7, val_ratio=0.15, _test_ratio=0.15):
    data_dir = Path(data_dir)
    image_files = sorted(data_dir.glob("*_img.nii*"))

    all_images = []
    all_masks = []

    for img_file in image_files:
        mask_file = img_file.parent / img_file.name.replace('_img', '_mask')
        if not mask_file.exists():
            continue
        all_images.append(img_file)
        all_masks.append(mask_file)

    np.random.seed(42)
    indices = np.random.permutation(len(all_images))

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_images = [all_images[i] for i in train_indices]
    train_masks = [all_masks[i] for i in train_indices]
    val_images = [all_images[i] for i in val_indices]
    val_masks = [all_masks[i] for i in val_indices]
    test_images = [all_images[i] for i in test_indices]
    test_masks = [all_masks[i] for i in test_indices]

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def run_experiment(experiment_name, use_residual=True, weighted=True, ce_weight=1.0, dice_weight=1.0, num_epochs=100, use_sens=True, use_dist=True):
    set_seed(SEED)

    data_dir = Path(r"./7013610/data/data")
    batch_size = 1
    learning_rate = 1e-3
    num_workers = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_images, train_masks, val_images, val_masks, test_images, test_masks = prepare_data_lists(data_dir)

    if len(train_images) == 0:
        print("Error: No data found!")
        return None

    train_dataset = MedicalImageDataset(train_images, train_masks)
    val_dataset = MedicalImageDataset(val_images, val_masks)
    test_dataset = MedicalImageDataset(test_images, test_masks)

    collate_fn = collate_fn_single if batch_size == 1 else collate_fn_pad

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_fn)

    num_classes = len(LABELS)
    model = build_network(num_input_channels=1, num_classes=num_classes, use_residual=use_residual)
    model = model.to(device)

    sens_matrix = weighted_matrices.create_sens_matrix(weighted_matrices.SENSITIVITY_RANKINGS) if use_sens else None
    dist_matrix = weighted_matrices.create_dist_matrix(weighted_matrices.AVG_DISTANCES) if use_dist else None
    weight_matrix = weighted_matrices.create_weighted_matrix(sens_matrix=sens_matrix, dist_matrix=dist_matrix, sens_importance=0.9)
    weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        weight_matrix=weight_matrix,
        learning_rate=learning_rate,
        max_epochs=num_epochs,
        weighted=weighted,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        experiment_name=experiment_name
    )

    return trainer.train(num_epochs=num_epochs)


def main():
    num_epochs = 100
    results = {}

    # Experiment 1: ResidualEncoderUNet with weighted=True, ce_weight=1.0
    results['resnet_weighted_ce1'] = run_experiment(
        experiment_name='resnet_weighted_ce1',
        use_residual=True,
        weighted=True,
        ce_weight=1.0,
        dice_weight=1.0,
        num_epochs=num_epochs
    )

    # Experiment 2: ResidualEncoderUNet with weighted=True, ce_weight=0 (dice only)
    results['resnet_weighted_ce0'] = run_experiment(
        experiment_name='resnet_weighted_ce0',
        use_residual=True,
        weighted=True,
        ce_weight=0.0,
        dice_weight=1.0,
        num_epochs=num_epochs
    )

    # Experiment 3: ResidualEncoderUNet with weighted=False (baseline)
    results['resnet_unweighted'] = run_experiment(
        experiment_name='resnet_unweighted',
        use_residual=True,
        weighted=False,
        ce_weight=1.0,
        dice_weight=1.0,
        num_epochs=num_epochs
    )

    # Experiment 4: ResidualEncoderUNet with only sens_matrix (no dist_matrix)
    results['resnet_sens_only'] = run_experiment(
        experiment_name='resnet_sens_only',
        use_residual=True,
        weighted=True,
        ce_weight=1.0,
        dice_weight=1.0,
        num_epochs=num_epochs,
        use_sens=True,
        use_dist=False
    )

    # Experiment 5: ResidualEncoderUNet with only dist_matrix (no sens_matrix)
    results['resnet_dist_only'] = run_experiment(
        experiment_name='resnet_dist_only',
        use_residual=True,
        weighted=True,
        ce_weight=1.0,
        dice_weight=1.0,
        num_epochs=num_epochs,
        use_sens=False,
        use_dist=True
    )

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for name, res in results.items():
        if res:
            print(f"\n{name}:")
            print(f"  Mean FG Dice: {res['mean_foreground_dice']:.4f}")
            print(f"  Mean Weighted FG Dice: {res['mean_weighted_foreground_dice']:.4f}")


if __name__ == "__main__":
    main()
