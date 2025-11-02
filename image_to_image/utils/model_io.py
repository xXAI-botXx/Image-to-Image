"""
Module to handle the input/output of PyTorch models.
In this case that means loading and saving models.


Functions:
- save_checkpoint
- get_single_model
- get_model
- load_model_weights
- load_and_get_model

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import os

import torch

import prime_printer as prime

from ..models.pix2pix import Pix2Pix
from ..models.resfcn import ResFCN
from ..models.residual_design_model import ResidualDesignModel
from ..models.transformer import PhysicFormer



# ---------------------------
#         > Saving <
# ---------------------------
def save_checkpoint(args, model, optimizer, scheduler, epoch, path='ckpt.pth'):
    """
    Saves a training checkpoint containing model, optimizer, and scheduler states.

    Parameter:
    - model (nn.Module):
        Model to save.
    - optimizer:
        Optimizer or list/tuple of optimizers.
    - scheduler:
        Scheduler or list/tuple of schedulers.
    - epoch (int):
        Current epoch index.
    - path (str):
        File path to save checkpoint to ('.pth' extension added automatically).

    Returns:
    - None
    """
    if not path.endswith(".pth"):
        path += ".pth"

    # set content to save
    checkpoint_saving = {'epoch': epoch, 'model_state': model.state_dict()}

    if isinstance(optimizer, (list, tuple)):
        for idx, cur_optimizer in enumerate(optimizer):
            checkpoint_saving[f'optim_state_{idx}'] = cur_optimizer.state_dict()
    else:
        checkpoint_saving[f'optim_state'] = optimizer.state_dict()

    if isinstance(scheduler, (list, tuple)):
        for idx, cur_scheduler in enumerate(scheduler):
            checkpoint_saving[f'sched_state_{idx}'] = cur_scheduler.state_dict()
    else:
        checkpoint_saving[f'sched_state'] = scheduler.state_dict()

    checkpoint_saving['args'] = args

    # save checkpoint
    torch.save(checkpoint_saving, path)

    # save info txt
    root_path, name = os.path.split(path)
    info_name = ".".join(name.split(".")[:-1]) + ".txt"
    with open(os.path.join(root_path, info_name), "w") as f:
        f.write(f"Last Model saved in epoch: {epoch}, at: {prime.get_time(pattern='DAY.MONTH.YEAR HOUR:MINUTE O\'Clock', time_zone='Europe/Berlin')}")



# ---------------------------
#        > Loading <
# ---------------------------
def get_single_model(model_name, args, criterion, device):
    """
    Returns a single model instance based on provided arguments.

    Supported models:
    - ResFCN
    - Pix2Pix
    - ResidualDesignModel
    - PhysicFormer

    Parameter:
    - model_name (str):
        Name of the model to initialize.
    - args:
        Parsed command-line arguments.
    - criterion:
        Criterion for Pix2Pixs second loss. Required during model initialization.
    - device:
        Target device (GPU or CPU) on which to place the model.

    Returns:
    - model (nn.Module): 
        Instantiated PyTorch model on the given device.
    """
    model_name = model_name.lower()

    if model_name== "resfcn":
        model = ResFCN(input_channels=args.resfcn_in_channels, 
                       hidden_channels=args.resfcn_hidden_channels, 
                       output_channels=args.resfcn_out_channels,
                         num_blocks=args.resfcn_num_blocks).to(device)
    elif model_name == "resfcn_2":
        model = ResFCN(input_channels=args.resfcn_2_in_channels, 
                       hidden_channels=args.resfcn_2_hidden_channels, 
                       output_channels=args.resfcn_2_out_channels, 
                       num_blocks=args.resfcn_2_num_blocks).to(device)
    elif model_name == "pix2pix":
        model = Pix2Pix(input_channels=args.pix2pix_in_channels, 
                        output_channels=args.pix2pix_out_channels, 
                        hidden_channels=args.pix2pix_hidden_channels, 
                        second_loss=criterion, 
                        lambda_second=args.pix2pix_second_loss_lambda).to(device)
    elif model_name == "pix2pix_2":
        model = Pix2Pix(input_channels=args.pix2pix_2_in_channels, 
                        output_channels=args.pix2pix_2_out_channels, 
                        hidden_channels=args.pix2pix_2_hidden_channels, 
                        second_loss=criterion, 
                        lambda_second=args.pix2pix_2_second_loss_lambda).to(device)
    elif model_name == "physicsformer":
        model = PhysicFormer(input_channels=args.physicsformer_in_channels, 
                             output_channels=args.physicsformer_out_channels, 
                             img_size=args.physicsformer_img_size, 
                             patch_size=args.physicsformer_patch_size, 
                             embedded_dim=args.physicsformer_embedded_dim, 
                             num_blocks=args.physicsformer_num_blocks,
                             heads=args.physicsformer_heads, 
                             mlp_dim=args.physicsformer_mlp_dim, 
                             dropout=args.physicsformer_dropout,
                             is_train=True if args.mode == "train" else False).to(device)
    elif model_name == "physicsformer_2":
        model = PhysicFormer(input_channels=args.physicsformer_in_channels_2, 
                             output_channels=args.physicsformer_out_channels_2, 
                             img_size=args.physicsformer_img_size_2, 
                             patch_size=args.physicsformer_patch_size_2, 
                             embedded_dim=args.physicsformer_embedded_dim_2, 
                             num_blocks=args.physicsformer_num_blocks_2,
                             heads=args.physicsformer_heads_2, 
                             mlp_dim=args.physicsformer_mlp_dim_2, 
                             dropout=args.physicsformer_dropout_2,
                             is_train=True if args.mode == "train" else False).to(device)
    else:
        raise ValueError(f"'{model_name}' is not a supported model.")

    return model



def get_model(args, device, criterion=None):
    """
    Returns a model object with the given args. Also complex models with sub-models can be loaded.
    This is the main function to get a model object.

    - args (argparse.ArgumentParser):
        Arguments to get information about the model class/object.
    - device (torch.device): 
        Device on which the model should get move to.
    - criterion (torch.nn.modules.loss._Loss, default=None): 
        Loss function. Some models save the loss internally. Not needed for inference only.

    Returns:
    - torch.nn.Module: 
        Loaded Model object without weights.
    """
    if args.model.lower() == "residual_design_model":
        model = ResidualDesignModel(base_model=get_single_model(model_name=args.base_model, args=args, criterion=criterion[0], device=device).to(device),
                                    complex_model=get_single_model(model_name=args.complex_model+"_2", args=args, criterion=criterion[1], device=device).to(device),
                                    combine_mode=args).to(device)
    else:
        model = get_single_model(model_name=args.model, args=args, criterion=criterion, device=device).to(device)

    return model



def load_model_weights(model_params_path):
    """
    Loads model weights (model states) from a pth file.

    Parameter:
    - model_params_path (str): 
        The path to the model parameters.

    Returns:
    - dict: 
        Loaded weights
    """
    return torch.load(model_params_path, weights_only=False)



# python -c "import torch;print(type(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))"
# <class 'torch.device'>
# python -c "import torch.nn as nn;print(isinstance(nn.MSELoss(), nn.modules.loss._Loss))"
def load_and_get_model(model_params_path, device, criterion=None):
    """
    Loads model weights (model states) from a pth file.
    And creates the right model object and return the model object with the weights loaded.

    Parameter:
    - model_params_path (str): 
        The path to the model parameters.
    - device (torch.device): 
        Device on which the model should get move to.
    - criterion (torch.nn.modules.loss._Loss, default=None): 
        Loss function. Some models save the loss internally. Not needed for inference only.

    Returns:
    - torch.nn.Module: 
        Loaded Model object with weights.
    - argparse.ArgumentParser: 
        Loaded arguments.
    """
    state_dict = load_model_weights(model_params_path)
    args = state_dict["args"]
    model = get_model(args=args, device=device, criterion=criterion)
    model.load_state_dict(state_dict["model_state"])
    model.eval()
    return model, args


