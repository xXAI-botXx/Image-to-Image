"""
Module to train, validate, and evaluate image-to-image models. 
Supports multiple models, losses, optimizers, schedulers, mixed-precision training,
checkpointing, and experiment tracking with MLflow and TensorBoard.

The train function handles full experiment orchestration including:
- Argument parsing and device setup.
- Dataset and dataloader initialization.
- Model, optimizer, loss function, and scheduler setup.
- Mixed precision (AMP) and warm-up handling.
- MLflow and TensorBoard logging.
- Periodic validation and checkpoint saving.

Functions:
- get_loss
- get_optimizer
- get_scheduler
- backward_model
- train_one_epoch
- evaluate
- train

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import os
import shutil
import time
import copy

import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision

from lion_pytorch import Lion

# Experiment tracking
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter

import prime_printer as prime


from ..utils.argument_parsing import parse_args
from ..utils.model_io import get_model, save_checkpoint
from ..utils.diffusion import add_noise_step

from ..data.physgen import PhysGenDataset
from ..data.residual_physgen import PhysGenResidualDataset, to_device

from ..models.resfcn import ResFCN
from ..models.pix2pix import Pix2Pix
from ..models.residual_design_model import ResidualDesignModel
from ..models.transformer import PhysicFormer
from ..models.uvit import UViT

from ..losses.weighted_combined_loss import WeightedCombinedLoss
from ..scheduler.warm_up import WarmUpScheduler
from..amp.dummy_scaler import DummyScaler



# ---------------------------
#      > Train Helpers <
# ---------------------------
def get_data(args):
    if args.model.lower() == "residual_design_model":
        train_dataset = PhysGenResidualDataset(variation=args.data_variation, mode="train", 
                                               fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size,
                                               reflexion_channels=args.reflexion_channels, reflexion_steps=args.reflexion_steps, reflexions_as_channels=args.reflexions_as_channels)
        
        val_dataset = PhysGenDataset(variation=args.data_variation, mode="validation", input_type="osm", output_type="standard", 
                                    fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size,
                                    reflexion_channels=args.reflexion_channels, reflexion_steps=args.reflexion_steps, reflexions_as_channels=args.reflexions_as_channels)
    else:
        train_dataset = PhysGenDataset(variation=args.data_variation, mode="train", input_type=args.input_type, output_type=args.output_type, 
                                    fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size,
                                    reflexion_channels=args.reflexion_channels, reflexion_steps=args.reflexion_steps, reflexions_as_channels=args.reflexions_as_channels)
        
        val_dataset = PhysGenDataset(variation=args.data_variation, mode="validation", input_type=args.input_type, output_type=args.output_type, 
                                    fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size,
                                    reflexion_channels=args.reflexion_channels, reflexion_steps=args.reflexion_steps, reflexions_as_channels=args.reflexions_as_channels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    return train_dataset, val_dataset, train_loader, val_loader



def get_loss(loss_name, args):
    """
    Returns a loss function instance based on the provided loss name.

    Supported losses:
    - L1 / L1_2
    - CrossEntropy / CrossEntropy_2
    - WeightedCombined / WeightedCombined_2

    Parameter:
    - loss_name (str):
        Name of the loss function.
    - args:
        Parsed command-line arguments with configured loss weights.

    Returns:
    - criterion (nn.Module): Instantiated loss function.
    """
    loss_name = loss_name.lower()

    if loss_name == "l1":
        criterion = nn.L1Loss()
    elif loss_name == "l1_2":
        criterion = nn.L1Loss()
    elif loss_name == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "crossentropy_2":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "weighted_combined":
        criterion = WeightedCombinedLoss( 
                        silog_lambda=args.wc_loss_silog_lambda, 
                        weight_silog=args.wc_loss_weight_silog, 
                        weight_grad=args.wc_loss_weight_grad,
                        weight_ssim=args.wc_loss_weight_ssim,
                        weight_edge_aware=args.wc_loss_weight_edge_aware,
                        weight_l1=args.wc_loss_weight_l1,
                        weight_var=args.wc_loss_weight_var,
                        weight_range=args.wc_loss_weight_range,
                        weight_blur=args.wc_loss_weight_blur
                    )    
    elif loss_name == "weighted_combined_2":
        criterion = WeightedCombinedLoss( 
                        silog_lambda=args.wc_loss_silog_lambda_2, 
                        weight_silog=args.wc_loss_weight_silog_2, 
                        weight_grad=args.wc_loss_weight_grad_2,
                        weight_ssim=args.wc_loss_weight_ssim_2,
                        weight_edge_aware=args.wc_loss_weight_edge_aware_2,
                        weight_l1=args.wc_loss_weight_l1_2,
                        weight_var=args.wc_loss_weight_var_2,
                        weight_range=args.wc_loss_weight_range_2,
                        weight_blur=args.wc_loss_weight_blur_2
                    )    
    else:
        raise ValueError(f"'{loss_name}' is not a supported loss.")
    
    return criterion



def get_optimizer(optimizer_name, model, lr, args):
    """
    Returns an optimizer for the given model.

    Supported optimizers:
    - Adam
    - AdamW

    Parameter:
    - optimizer_name (str):
        Name of the optimizer.
    - model (nn.Module):
        Model whose parameters should be optimized.
    - lr (float):
        Learning rate.
    - args:
        Parsed command-line arguments with optimizer configuration.

    Returns:
    - optimizer (torch.optim.Optimizer): Instantiated optimizer.
    """
    optimizer_name = optimizer_name.lower()

    weight_decay_rate = args.weight_decay_rate if args.weight_decay else 0

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,  betas=(0.9, 0.95), weight_decay=weight_decay_rate)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr,  betas=(0.9, 0.95), weight_decay=weight_decay_rate)
    elif optimizer_name == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=lr,  weight_decay=weight_decay_rate)
    elif optimizer_name == "lion":
        optimizer = Lion(model.parameters(), lr=lr,  weight_decay=weight_decay_rate)
    else:
        raise ValueError(f"'{optimizer_name}' is not a supported optimizer.")
    
    return optimizer



def get_scheduler(scheduler_name, optimizer, args):
    """
    Returns a learning rate scheduler for the given optimizer.

    Supported schedulers:
    - StepLR
    - CosineAnnealingLR

    Parameter:
    - scheduler_name (str):
        Name of the scheduler.
    - optimizer:
        Optimizer whose learning rate will be managed.
    - args:
        Parsed command-line arguments containing scheduler configuration.

    Returns:
    - scheduler (torch.optim.lr_scheduler): Instantiated scheduler.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(f"'{scheduler_name}' is not a supported scheduler.")
    
    return scheduler



def get_params(args, device, n_model_params, current_save_name, checkpoint_save_dir):
    return {
            # General
            "mode": args.mode,
            "device": str(device),

            # Training
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss_function": args.loss,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "weight_decay_rate": args.weight_decay_rate,
            "gradient_clipping": args.gradient_clipping,
            "gradient_clipping_threshold": args.gradient_clipping_threshold,
            "scheduler": args.scheduler,
            "use_warm_up": args.use_warm_up,
            "warm_up_start_lr": args.warm_up_start_lr,
            "warm_up_step_duration": args.warm_up_step_duration,
            "use_amp": args.activate_amp,
            "amp_scaler": args.amp_scaler,
            "save_only_best_model": args.save_only_best_model,

            # Loss
            "wc_loss_silog_lambda": args.wc_loss_silog_lambda,
            "wc_loss_weight_silog": args.wc_loss_weight_silog,
            "wc_loss_weight_grad": args.wc_loss_weight_grad,
            "wc_loss_weight_ssim": args.wc_loss_weight_ssim,
            "wc_loss_weight_edge_aware": args.wc_loss_weight_edge_aware,
            "wc_loss_weight_l1": args.wc_loss_weight_l1,
            "wc_loss_weight_var": args.wc_loss_weight_var,
            "wc_loss_weight_range": args.wc_loss_weight_range,
            "wc_loss_weight_blur": args.wc_loss_weight_blur,

            # Model
            "model": args.model,
            "n_model_params": n_model_params,
            "resfcn_in_channels": args.resfcn_in_channels,
            "resfcn_hidden_channels": args.resfcn_hidden_channels,
            "resfcn_out_channels": args.resfcn_out_channels,
            "resfcn_num_blocks": args.resfcn_num_blocks,

            "pix2pix_in_channels": args.pix2pix_in_channels,
            "pix2pix_hidden_channels": args.pix2pix_hidden_channels,
            "pix2pix_out_channels": args.pix2pix_out_channels,
            "pix2pix_second_loss_lambda": args.pix2pix_second_loss_lambda,

            "physicsformer_in_channels": args.physicsformer_in_channels,
            "physicsformer_out_channels": args.physicsformer_out_channels,
            "physicsformer_img_size": args.physicsformer_img_size,
            "physicsformer_patch_size": args.physicsformer_patch_size,
            "physicsformer_embedded_dim": args.physicsformer_embedded_dim,
            "physicsformer_num_blocks": args.physicsformer_num_blocks,
            "physicsformer_heads": args.physicsformer_heads,
            "physicsformer_mlp_dim": args.physicsformer_mlp_dim,
            "physicsformer_dropout": args.physicsformer_dropout,

            "uvit_in_channels": args.uvit_in_channels, 
            "uvit_hidden_channels": args.uvit_hidden_channels,
            "uvit_out_channels": args.uvit_out_channels,
            "uvit_image_size": args.uvit_image_size,
            "uvit_timesteps": args.uvit_timesteps,

            # Data
            "data_variation": args.data_variation,
            "input_type": args.input_type,
            "output_type": args.output_type,
            "fake_rgb_output": args.fake_rgb_output,
            "make_14_dividable_size": args.make_14_dividable_size,

            # Experiment tracking
            "experiment_name": args.experiment_name,
            "run_name": current_save_name, # CURRENT_SAVE_NAME,
            "tensorboard_path": args.tensorboard_path,
            "save_path": args.save_path,
            "checkpoint_save_dir": checkpoint_save_dir,
            "cmap": args.cmap,

            # >> Residual Model <<
            "base_model": args.base_model,
            "complex_model": args.complex_model,
            "combine_mode": args.combine_mode,

            # ---- Loss (2nd branch)
            "loss_2": args.loss_2,
            "wc_loss_silog_lambda_2": args.wc_loss_silog_lambda_2,
            "wc_loss_weight_silog_2": args.wc_loss_weight_silog_2,
            "wc_loss_weight_grad_2": args.wc_loss_weight_grad_2,
            "wc_loss_weight_ssim_2": args.wc_loss_weight_ssim_2,
            "wc_loss_weight_edge_aware_2": args.wc_loss_weight_edge_aware_2,
            "wc_loss_weight_l1_2": args.wc_loss_weight_l1_2,
            "wc_loss_weight_var_2": args.wc_loss_weight_var_2,
            "wc_loss_weight_range_2": args.wc_loss_weight_range_2,
            "wc_loss_weight_blur_2": args.wc_loss_weight_blur_2,

            # ---- ResFCN Model 2
            "resfcn_2_in_channels": args.resfcn_2_in_channels,
            "resfcn_2_hidden_channels": args.resfcn_2_hidden_channels,
            "resfcn_2_out_channels": args.resfcn_2_out_channels,
            "resfcn_2_num_blocks": args.resfcn_2_num_blocks,

            # ---- Pix2Pix Model 2
            "pix2pix_2_in_channels": args.pix2pix_2_in_channels,
            "pix2pix_2_hidden_channels": args.pix2pix_2_hidden_channels,
            "pix2pix_2_out_channels": args.pix2pix_2_out_channels,
            "pix2pix_2_second_loss_lambda": args.pix2pix_2_second_loss_lambda,

            # ---- PhysicsFormer Model 2
            "physicsformer_in_channels_2": args.physicsformer_in_channels_2,
            "physicsformer_out_channels_2": args.physicsformer_out_channels_2,
            "physicsformer_img_size_2": args.physicsformer_img_size_2,
            "physicsformer_patch_size_2": args.physicsformer_patch_size_2,
            "physicsformer_embedded_dim_2": args.physicsformer_embedded_dim_2,
            "physicsformer_num_blocks_2": args.physicsformer_num_blocks_2,
            "physicsformer_heads_2": args.physicsformer_heads_2,
            "physicsformer_mlp_dim_2": args.physicsformer_mlp_dim_2,
            "physicsformer_dropout_2": args.physicsformer_dropout_2,
        
            # ---- U-ViT Model 2
            "uvit_2_in_channels": args.uvit_2_in_channels,
            "uvit_2_hidden_channels": args.uvit_2_hidden_channels,
            "uvit_2_out_channels": args.uvit_2_out_channels,
            "uvit_2_image_size": args.uvit_2_image_size,
            "uvit_2_timesteps": args.uvit_2_timesteps,

        }


def backward_model(model, x, y, optimizer, criterion, device, epoch, amp_scaler, gradient_clipping_threshold=None):
    """
    Performs a backward pass, optimizer step, and mixed-precision handling.

    Parameter:
    - model (nn.Module):
        Model to train.
    - x (torch.Tensor):
        Input tensor.
    - y (torch.Tensor):
        Target tensor.
    - optimizer:
        Optimizer or tuple of optimizers (for Pix2Pix/ResidualDesignModel).
    - criterion:
        Loss function.
    - device (torch.device):
        Device to perform computation on.
    - epoch (int):
        Current training epoch.
    - amp_scaler (GradScaler):
        Gradient scaler for mixed-precision training.
    - gradient_clipping_threshold (float, optional):
        Maximum allowed gradient norm for clipping.

    Returns:
    - loss (torch.Tensor): Computed loss value for the batch.
    """
    if isinstance(model, Pix2Pix):
        if epoch is None or epoch % 2 == 0:
            model.discriminator_step(x, y, optimizer[1], amp_scaler, device, gradient_clipping_threshold)
        loss, _, _ = model.generator_step(x, y, optimizer[0], amp_scaler, device, gradient_clipping_threshold)
    elif isinstance(model, UViT):
        optimizer.zero_grad()

        t = torch.randint(0, model.timesteps, (y.size(0),), device=device).long()

        # add noise to the target
        noise = torch.randn_like(y)
        noisy_y = add_noise_step(y, t, model.schedule_alphas_cumprod, noise)

        # add input image as additional channel
        noisy_y_with_x = torch.cat([noisy_y, x], dim=1)  # shape [B, 2, H, W]

        # predict noise
        with autocast(device_type=device.type):
            noise_pred = model(noisy_y_with_x, t, inference=False)
            # loss = nn.MSELoss()(noise_pred, noise)
            loss = criterion(noise_pred, noise)

        # backpropagate + rescaling
        if amp_scaler:
            amp_scaler.scale(loss).backward()
            if gradient_clipping_threshold:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_threshold)

            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            if gradient_clipping_threshold:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_threshold)
            optimizer.step()
    elif amp_scaler:
        # reset gradients 
        optimizer.zero_grad()

        with autocast(device_type=device.type):
            y_predict = model(x)
            loss = criterion(y_predict, y)
        amp_scaler.scale(loss).backward()
        if gradient_clipping_threshold:
            # Unscale first!
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_threshold)
        amp_scaler.step(optimizer)
        amp_scaler.update()
    else:
        # reset gradients 
        optimizer.zero_grad()

        y_predict = model(x)
        loss = criterion(y_predict, y)
        loss.backward()
        if gradient_clipping_threshold:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_threshold)
        optimizer.step()
    return loss



def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None, amp_scaler=None, gradient_clipping_threshold=None):
    """
    Runs one full epoch of training and returns the average loss.

    Parameter:
    - model (nn.Module):
        Model to train.
    - loader (DataLoader):
        Data loader containing training batches.
    - optimizer:
        Optimizer or tuple of optimizers.
    - criterion:
        Loss function or tuple of losses (for multi-stage models).
    - device (torch.device):
        Device to use for training.
    - epoch (int, optional):
        Current epoch index.
    - amp_scaler (GradScaler, optional):
        Mixed-precision scaler.
    - gradient_clipping_threshold (float, optional):
        Max allowed gradient norm.

    Returns:
    - avg_loss (float): Average training loss for the epoch.
    """
    # change to train mode -> calc gradients
    model.train()

    total_loss = 0.0
    # for x, y in tqdm(loader, desc=f"Epoch {epoch:03}", leave=True, ascii=True mininterval=5):
    for x, y in loader:

        if not isinstance(model, ResidualDesignModel):
            x, y = x.to(device), y.to(device)

        if isinstance(model, ResidualDesignModel):
            base_input, complex_input = x
            base_target, complex_target, target_= y

            # Basline
            base_input = base_input.to(device)
            base_target = base_target.to(device)
            base_loss = backward_model(model=model.base_model, x=base_input, y=base_target, optimizer=optimizer[0], criterion=criterion[0], device=device, epoch=epoch, amp_scaler=amp_scaler, gradient_clipping_threshold=gradient_clipping_threshold)
            # del base_input
            # torch.cuda.empty_cache()

            # Complex
            complex_input = complex_input.to(device)
            complex_target = complex_target.to(device)
            complex_loss = backward_model(model=model.complex_model, x=complex_input, y=complex_target, optimizer=optimizer[1], criterion=criterion[1], device=device, epoch=epoch, amp_scaler=amp_scaler, gradient_clipping_threshold=gradient_clipping_threshold)
            # del complex_input
            # torch.cuda.empty_cache()

            # Fusion
            target_ = target_.to(device)
            combine_loss = model.combine_net.backward(base_target, complex_target, target_)

            model.backward(base_target, complex_target, target_)

            model.last_base_loss = base_loss
            model.last_complex_loss = complex_loss
            model.last_combined_loss = combine_loss
            loss = base_loss + complex_loss + combine_loss
        else:
            loss = backward_model(model=model, x=x, y=y, optimizer=optimizer, criterion=criterion, device=device, epoch=epoch, amp_scaler=amp_scaler, gradient_clipping_threshold=gradient_clipping_threshold)
        total_loss += loss.item()
    return total_loss / len(loader)



@torch.no_grad()
def evaluate(model, loader, criterion, device, writer=None, epoch=None, save_path=None, cmap="gray", use_tqdm=True):
    """
    Evaluates the model on the validation set and logs results to TensorBoard or MLflow.

    Parameter:
    - model (nn.Module):
        Model to evaluate.
    - loader (DataLoader):
        Validation data loader.
    - criterion:
        Loss function used for evaluation.
    - device (torch.device):
        Device to perform inference on.
    - writer (SummaryWriter, optional):
        TensorBoard writer for visualization.
    - epoch (int, optional):
        Current epoch index for logging.
    - save_path (str, optional):
        Directory to save sample images.
    - cmap (str):
        Colormap for saved images.
    - use_tqdm (bool):
        Whether to use tqdm progress bar.

    Returns:
    - avg_loss (float): Average validation loss.
    """
    model.eval()

    total_loss = 0.0
    is_first_round = True

    if use_tqdm:
        validation_iter = tqdm(loader, desc="Validation", ascii=True, mininterval=3)
    else:
        validation_iter = loader

    for x, y in validation_iter:
        x, y = x.to(device), y.to(device)
        y_predict = model(x)
        total_loss += criterion(y_predict, y).item()

        if is_first_round:
            if writer:
                # Convert to grid
                img_grid_input = torchvision.utils.make_grid(x[:4].cpu(), normalize=True, scale_each=True)
                max_val = max(y_predict.max().item(), 1e-8)
                if y_predict.ndim == 3:  # e.g., [B, H, W]
                    img_grid_pred = torchvision.utils.make_grid(y_predict[:4].unsqueeze(1).float().cpu() / max_val)
                else:  # [B, 1, H, W]
                    img_grid_pred = torchvision.utils.make_grid(y_predict[:4].float().cpu() / max_val)

                max_val = max(y.max().item(), 1e-8)
                if y.ndim == 3:
                    img_grid_gt = torchvision.utils.make_grid(y[:4].unsqueeze(1).float().cpu() / max_val)
                else:
                    img_grid_gt = torchvision.utils.make_grid(y[:4].float().cpu() / max_val)

                # Log to TensorBoard
                writer.add_image("Input", img_grid_input, epoch)
                writer.add_image("Prediction", img_grid_pred, epoch)
                writer.add_image("GroundTruth", img_grid_gt, epoch)

            if save_path:
                # os.makedirs(save_path, exist_ok=True)
                prediction_path = os.path.join(save_path, f"{epoch}_prediction.png")
                plt.imsave(prediction_path, 
                        y_predict[0].detach().cpu().numpy().squeeze(), cmap=cmap)
                mlflow.log_artifact(prediction_path, artifact_path="images")

                input_path = os.path.join(save_path, f"{epoch}_input.png")
                plt.imsave(input_path, 
                        x[0][0].detach().cpu().numpy().squeeze(), cmap=cmap)
                mlflow.log_artifact(input_path, artifact_path="images")

                ground_truth_path = os.path.join(save_path, f"{epoch}_ground_truth.png")
                plt.imsave(ground_truth_path, 
                        y[0].detach().cpu().numpy().squeeze(), cmap=cmap)
                mlflow.log_artifacts(ground_truth_path, artifact_path="images")

                # alternative direct save:
                # import numpy as np
                # import io
                # from PIL import Image

                # img = (y[0].detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
                # img_pil = Image.fromarray(img)

                # buf = io.BytesIO()
                # img_pil.save(buf, format='PNG')
                # buf.seek(0)

                # mlflow.log_image(image=img_pil, artifact_file=f"images/{epoch}_ground_truth.png")
                
        is_first_round = False

    return total_loss / len(loader)



# ---------------------------
#        > Train Main <
# ---------------------------
def train(args=None):
    """
    Main training loop for image-to-image tasks.

    Workflow:
    1. Initializes the training and validation datasets based on model type.
    2. Constructs the model and its loss functions.
    3. Configures optimizers, learning rate schedulers, and optional warm-up phases.
    4. Enables mixed precision (AMP) if selected.
    5. Sets up MLflow experiment tracking and TensorBoard visualization.
    6. Executes the epoch loop:
        - Trains the model for one epoch (`train_one_epoch()`).
        - Optionally evaluates on the validation set.
        - Logs metrics and learning rates.
        - Updates the scheduler.
        - Saves checkpoints (best or periodic).
    7. Logs the trained model and experiment results to MLflow upon completion.

    Parameters:
    - args : argparse.Namespace, optional
        Parsed command-line arguments containing all training configurations.
        If None, the function will automatically call `parse_args()` to obtain them.
    
    Returns:
    - None: The function performs training and logging in-place without returning values.

    Notes:
    - Automatically handles model-specific configurations (e.g., Pix2Pix discriminator, ResidualDesignModel branches).
    - Uses `prime.get_time()` to generate time-stamped run names.
    - Supports gradient clipping and various learning rate schedulers.

    Logging:
    - **MLflow**: Stores metrics, hyperparameters, checkpoints, and final model.
    - **TensorBoard**: Logs training/validation losses, learning rates, and sub-loss components.
    """
    print("\n---> Welcome to Image-to-Image Training <---")

    print("\nChecking your Hardware:")
    print(prime.get_hardware())
    print("\n\n")

    # Parse arguments
    if args is None:
        args = parse_args()

    CURRENT_SAVE_NAME = prime.get_time(pattern="YEAR-MONTH-DAY_HOUR_MINUTE_", time_zone="Europe/Berlin") + args.run_name

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset loading
    train_dataset, val_dataset, train_loader, val_loader = get_data(args)

    # Loss
    criterion = get_loss(loss_name=args.loss, args=args)
    
    if args.model.lower() == "residual_design_model":
        criterion = [
            criterion,
            get_loss(loss_name=args.loss_2+"_2", args=args)
        ]

    # Model Loading
    model = get_model(args, device, criterion=criterion)

    # get parameter amount
    n_model_params = 0
    for cur_model_param in model.parameters():
        n_model_params += cur_model_param.numel()

    INPUT_CHANNELS = model.get_input_channels()

    # Optimizer
    if args.model.lower() == "pix2pix":
        optimizer = [get_optimizer(optimizer_name=args.optimizer, model=model.generator, lr=args.lr, args=args),
                     get_optimizer(optimizer_name=args.optimizer_2, model=model.discriminator, lr=args.lr, args=args)]
    elif args.model.lower() == "residual_design_model":
        optimizer = [get_optimizer(optimizer_name=args.optimizer, model=model.base_model, lr=args.lr, args=args),
                     get_optimizer(optimizer_name=args.optimizer_2, model=model.complex_model, lr=args.lr, args=args)]
        
        if args.base_model.lower() == "pix2pix":
            optimizer[0] = [get_optimizer(optimizer_name=args.optimizer, model=model.base_model.generator, lr=args.lr, args=args), 
                            get_optimizer(optimizer_name=args.optimizer, model=model.base_model.discriminator, lr=args.lr, args=args)]
        if args.complex_model.lower() == "pix2pix":
            optimizer[1] = [get_optimizer(optimizer_name=args.optimizer_2, model=model.complex_model.generator, lr=args.lr, args=args),
                            get_optimizer(optimizer_name=args.optimizer_2, model=model.complex_model.discriminator, lr=args.lr, args=args)]
    else:
        optimizer = get_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr, args=args)

    # Scheduler
    if args.model.lower() == "residual_design_model":

        if args.base_model.lower() == "pix2pix":
            scheduler_1 = [get_scheduler(scheduler_name=args.scheduler, optimizer=optimizer[0][0], args=args),
                           get_scheduler(scheduler_name=args.scheduler, optimizer=optimizer[0][1], args=args)]
        else:
            scheduler_1 = get_scheduler(scheduler_name=args.scheduler, optimizer=optimizer[0], args=args)
            
        if args.complex_model.lower() == "pix2pix":
            scheduler_2 = [get_scheduler(scheduler_name=args.scheduler_2, optimizer=optimizer[1][0], args=args),
                           get_scheduler(scheduler_name=args.scheduler_2, optimizer=optimizer[1][1], args=args)]
        else:
            scheduler_2 = get_scheduler(scheduler_name=args.scheduler_2, optimizer=optimizer[1], args=args)
            
            scheduler = [scheduler_1, scheduler_2]
    elif args.model.lower() == "pix2pix":
        scheduler = [get_scheduler(scheduler_name=args.scheduler, optimizer=optimizer[0], args=args),
                     get_scheduler(scheduler_name=args.scheduler_2, optimizer=optimizer[1], args=args)]
    else:
        scheduler = get_scheduler(scheduler_name=args.scheduler, optimizer=optimizer, args=args)

    # Warm-Up Scheduler
    if args.use_warm_up:
        if isinstance(scheduler, (tuple, list)):
            new_scheduler = []
            for cur_scheduler in scheduler:
                new_scheduler += [WarmUpScheduler(start_lr=args.warm_up_start_lr, end_lr=args.lr, optimizer=cur_scheduler.optimizer, scheduler=cur_scheduler, step_duration=args.warm_up_step_duration)]

            scheduler = new_scheduler
        else:
            if scheduler is not None:
                cur_optimizer = scheduler.optimizer
            else:
                cur_optimizer = None
            scheduler = WarmUpScheduler(start_lr=args.warm_up_start_lr, end_lr=args.lr, optimizer=cur_optimizer, scheduler=scheduler, step_duration=args.warm_up_step_duration)

    # AMP Scaler
    if args.activate_amp == False:
        amp_scaler = None
    elif args.amp_scaler.lower() == "none":
        amp_scaler = DummyScaler()
    elif args.amp_scaler.lower() == "grad":
        amp_scaler = GradScaler()
    else:
        raise ValueError(f"'{args.amp_scaler}' is not an supported scaler.")

    # setup checkpoint saving
    checkpoint_save_dir = os.path.join(args.checkpoint_save_dir, args.experiment_name, CURRENT_SAVE_NAME)
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    shutil.rmtree(checkpoint_save_dir)
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    # setup intermediate image saving
    save_path = os.path.join(args.save_path, args.experiment_name, CURRENT_SAVE_NAME)
    os.makedirs(save_path, exist_ok=True)
    shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    # setup gradient clipping
    if args.gradient_clipping:
        gradient_clipping_threshold = args.gradient_clipping_threshold
    else:
        gradient_clipping_threshold = None

    mlflow.set_experiment(args.experiment_name)
    # same as:
        # mlflow.create_experiment(args.experiment_name)
        # mlflow.get_experiment_by_name(experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=CURRENT_SAVE_NAME):

        # TensorBoard writer
        tensorboard_path = os.path.join(args.tensorboard_path, args.experiment_name, CURRENT_SAVE_NAME)
        os.makedirs(tensorboard_path, exist_ok=True)
        shutil.rmtree(tensorboard_path)
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_path)

        # Log hyperparameters
        params =  get_params(args=args, 
                             device=device, 
                             n_model_params=n_model_params, 
                             current_save_name=CURRENT_SAVE_NAME, 
                             checkpoint_save_dir=checkpoint_save_dir)
        
        # mlflow.log_params(vars(args))
        mlflow.log_params(params)

        params_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
        writer.add_text("hyperparameters", params_text, 0)
                                                
        print("\n---------------------------\n     - RUN SETTINGS -\n"+params_text+"\n---------------------------\n\n")

        print(f"Train dataset size: {len(train_dataset)} | Validation dataset size: {len(val_dataset)}")
        mlflow.log_metrics({
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset)
        })


        # log model architecture
        #    -> create dataset for that
        if isinstance(INPUT_CHANNELS, (list, tuple)):
            dummy_inference_data = [torch.ones(size=(1, channels, 256, 256)).to(device) for channels in INPUT_CHANNELS]
        else:
            dummy_inference_data = torch.ones(size=(1, INPUT_CHANNELS, 256, 256)).to(device)
        if isinstance(model, UViT):
            original_forward = model.forward
            model.forward = lambda x: original_forward(x, dummy_pass=True)
        writer.add_graph(model, dummy_inference_data)
        if isinstance(model, UViT):
            model.forward = original_forward

        # Run Training
        last_best_loss = float("inf")
        try:
            epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", ascii=True)
            for epoch in epoch_iter:
                start_time = time.time()
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, amp_scaler, gradient_clipping_threshold)
                duration = time.time() - start_time
                if epoch % args.validation_interval == 0:
                    val_loss = evaluate(model, val_loader, criterion, device, writer=writer, epoch=epoch, save_path=save_path, cmap=args.cmap, use_tqdm=False)
                else:
                    val_loss = float("inf")

                val_str = f"{val_loss:.4f}" if epoch % args.validation_interval == 0 else "N/A"
                epoch_iter.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=val_str, time_needed=f"{duration:.2f}s")
                # tqdm.write(f" -> Train Loss: {train_loss:.4f} | Val Loss: {val_str} | Time: {duration:.2f}")
                # \n\n[Epoch {epoch:02}/{args.epochs}]
                
                # Hint: Tensorboard and mlflow does not like spaces in tags!

                # Log to TensorBoard
                writer.add_scalar("Time/epoch_duration", duration, epoch)
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                if isinstance(scheduler, list):
                    if isinstance(scheduler[0], list):
                        writer.add_scalar("LR/generator", scheduler[0][0].get_last_lr()[0], epoch)
                        writer.add_scalar("LR/discriminator", scheduler[0][1].get_last_lr()[0], epoch)
                    else:
                        name = "generator" if args.model.lower() == "pix2pix" else "base_model"
                        writer.add_scalar(f"LR/{name}", scheduler[0].get_last_lr()[0], epoch)

                    if isinstance(scheduler[1], list):
                        writer.add_scalar("LR/generator", scheduler[1][0].get_last_lr()[0], epoch)
                        writer.add_scalar("LR/discriminator", scheduler[1][1].get_last_lr()[0], epoch)
                    else:
                        name = "discriminator" if args.model.lower() == "pix2pix" else "complex_model"
                        writer.add_scalar(f"LR/{name}", scheduler[1].get_last_lr()[0], epoch)
                else:
                    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

                # Log to MLflow
                if type(scheduler) in [list, tuple]:
                    metrics = {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                    idx = 0
                    for idx, cur_scheduler in enumerate(scheduler):
                        metrics[f"lr_{idx}"] = cur_scheduler.get_last_lr()[0]
                    mlflow.log_metrics(metrics, step=epoch)
                else:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": scheduler.get_last_lr()[0]
                    }, step=epoch)

                # add sub losses / loss components
                if args.model.lower() in ["pix2pix", "residual_design_model"]:
                    losses = model.get_dict()
                    for name, value in losses.items():
                        writer.add_scalar(f"LossComponents/{name}", value, epoch)
                    mlflow.log_metrics(losses, step=epoch)

                if type(criterion) in [list, tuple]:
                    if args.loss in ["weighted_combined"]:
                        losses = criterion[0].get_dict()
                        for name, value in losses.items():
                            writer.add_scalar(f"LossComponents/{name}", value, epoch)
                        mlflow.log_metrics(losses, step=epoch)

                    if args.loss_2 in ["weighted_combined"] and args.model.lower() in ["residual_design_model"]:
                        losses = criterion[1].get_dict()
                        for name, value in losses.items():
                            writer.add_scalar(f"LossComponents/{name}", value, epoch)
                        mlflow.log_metrics(losses, step=epoch)
                else:
                    if args.loss in ["weighted_combined"]:
                        losses = criterion.get_dict()
                        for name, value in losses.items():
                            writer.add_scalar(f"LossComponents/{name}", value, epoch)
                        mlflow.log_metrics(losses, step=epoch)

                

                # Step scheduler
                if args.model.lower() in ["pix2pix"]:
                    scheduler[0].step()
                    scheduler[1].step()
                else:
                    scheduler.step()

                # Save Checkpoint
                if args.save_only_best_model:
                    if val_loss < last_best_loss or (last_best_loss == float("inf") and epoch == args.epochs):
                        last_best_loss = val_loss
                        checkpoint_path = os.path.join(checkpoint_save_dir, f"best_checkpoint.pth")
                        save_checkpoint(args, model, optimizer, scheduler, epoch, checkpoint_path)

                        # Log model checkpoint path
                        mlflow.log_artifact(checkpoint_path)
                elif epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                    checkpoint_path = os.path.join(checkpoint_save_dir, f"epoch_{epoch}.pth")
                    save_checkpoint(args, model, optimizer, scheduler, epoch, checkpoint_path)

                    # Log model checkpoint path
                    mlflow.log_artifact(checkpoint_path)

            # Log final model
            try:
                mlflow.pytorch.log_model(model.cpu(), name="model", input_example=dummy_inference_data.cpu().numpy())
            except Exception as e:
                print(e)
                mlflow.pytorch.log_model(model, name="model")
            mlflow.end_run()
        finally:
            writer.close()

        print("Training completed.")



if __name__ == "__main__":
    train()



