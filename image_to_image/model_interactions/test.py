"""
Module for testing image to image models.

Functions:
- test

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn

from ..utils.argument_parsing import parse_args
from ..utils.model_io import load_and_get_model

from .train import evaluate, get_loss  # reuse the evaluate function

from ..data.physgen import PhysGenDataset
from ..data.residual_physgen import PhysGenResidualDataset

from ..models.resfcn import ResFCN 


# ---------------------------
#        > Run Test <
# ---------------------------
def test(args=None):
    """
    Runs evaluation of a pre-trained ResFCN model on a test dataset.

    This function:
    - Loads the test dataset using `PhysGenDataset`.
    - Loads a ResFCN model with parameters specified in `args`.
    - Loads model weights from a checkpoint file.
    - Evaluates the model on the test dataset using a specified loss criterion.

    Parameters:
    - args (Namespace or None): Optional argument namespace, typically from argparse.
        Required fields in `args`:
        - device (str): Device for computation ("cuda" or "cpu").
        - data_variation (str): Dataset variation to use for testing.
        - input_type (str): Input type for dataset.
        - output_type (str): Output type for dataset.
        - fake_rgb_output (bool): Flag for dataset preprocessing.
        - make_14_dividable_size (bool): Flag for resizing dataset images.
        - batch_size (int): Batch size for DataLoader.
        - resfcn_in_channels (int): Number of input channels for the ResFCN model.
        - resfcn_hidden_channels (int): Number of hidden channels for ResFCN.
        - resfcn_out_channels (int): Number of output channels for ResFCN.
        - resfcn_num_blocks (int): Number of residual blocks in ResFCN.
        - model_params_path (str): Path to the saved model checkpoint.
        - loss (str): Loss type to use for evaluation ("l1" or "crossentropy").

    Returns:
    - None: Prints the test loss to stdout.
    """
    if args is None:
        args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset loading
    if args.model.lower() == "residual_design_model":
        test_dataset = PhysGenResidualDataset(variation=args.data_variation, mode="test", 
                                              fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size,
                                              reflexion_channels=args.reflexion_channels, reflexion_steps=args.reflexion_steps, reflexions_as_channels=args.reflexions_as_channels)
    else:
        test_dataset = PhysGenDataset(variation=args.data_variation, mode="test", input_type=args.input_type, output_type=args.output_type, 
                                      fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size,
                                      reflexion_channels=args.reflexion_channels, reflexion_steps=args.reflexion_steps, reflexions_as_channels=args.reflexions_as_channels)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model Loading
    model, _ = load_and_get_model(args.model_params_path, device, criterion=None)

    # Criterion for evaluation
    criterion = get_loss(args.loss, args)
    print(f"Used Loss: {args.loss}")

    # Run evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")



if __name__ == "__main__":
    test()


