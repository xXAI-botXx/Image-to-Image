"""
Module for inferencing image to image models.

Functions:
- load_image
- save_output
- inference

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import os

import cv2
from PIL import Image

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils.argument_parsing import parse_args
from ..data.physgen import PhysGenDataset
from ..models.resfcn import ResFCN 

# ---------------------------
#    > Inference Helper <
# ---------------------------
def load_image(path, width=256, height=256, grayscale=True):
    """
    Loads an image from disk, resizes it, converts to tensor, and adds a batch dimension.

    Parameters:
    - path (str): 
        Path to the image file.
    - width (int): 
        Desired image width after resizing (default=256).
    - height (int): 
        Desired image height after resizing (default=256).
    - grayscale (bool): 
        Whether to load the image as grayscale (default=True).

    Returns:
    - torch.Tensor: Image tensor with shape [1, C, H, W], ready for model input.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
    ])
    img = Image.fromarray(img)
    return transform(img).unsqueeze(0)  # Add batch dimension


def save_output(tensor, path):
    """
    Saves a tensor as an image file to disk.

    Parameters:
    - tensor (torch.Tensor): 
        Image tensor to save. Expected shape [B, C, H, W] or [C, H, W].
    - path (str): 
        Path where the image will be saved.
    """
    from torchvision.utils import save_image
    save_image(tensor, path)


# ---------------------------
#       > Inference <
# ---------------------------
def inference(args=None):
    """
    Runs inference using a pre-trained ResFCN model on a dataset or a custom image directory.

    The function supports:
    - Loading a test dataset if no custom images are provided.
    - Loading a pre-trained model checkpoint.
    - Running inference on either dataset or custom images.
    - Saving predicted outputs to the specified output directory.

    Parameters:
    - args (Namespace or None): Optional argument namespace, typically from argparse.
        Required fields in `args`:
        - device (str): Device for computation ("cuda" or "cpu").
        - image_dir_path (str or None): Path to custom images (optional).
        - output_dir (str): Directory where outputs will be saved.
        - data_variation (str): Dataset variation if using dataset.
        - input_type (str): Input type for dataset.
        - output_type (str): Output type for dataset.
        - fake_rgb_output (bool): Flag for dataset processing.
        - make_14_dividable_size (bool): Flag for resizing dataset images.
        - resfcn_in_channels (int): Number of input channels for the ResFCN model.
        - resfcn_hidden_channels (int): Number of hidden channels for ResFCN.
        - resfcn_out_channels (int): Number of output channels for ResFCN.
        - resfcn_num_blocks (int): Number of residual blocks in ResFCN.
        - model_params_path (str): Path to the saved model checkpoint.

    Returns:
    - None: The function saves predicted images to disk.
    """
    if args is None:
        args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    custom_images = not type(args.image_dir_path) == type(None) and os.path.exists(args.image_dir_path)

    # Data Loading
    if not custom_images:
        test_dataset = PhysGenDataset(variation=args.data_variation, mode="test", input_type=args.input_type, output_type=args.output_type, 
                                    fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size)
        test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Model Loading
    model = ResFCN(in_channels=args.resfcn_in_channels, hidden_channels=args.resfcn_hidden_channels, out_channels=args.resfcn_out_channels, num_blocks=args.resfcn_num_blocks).to(device)

    checkpoint = torch.load(args.model_params_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # run inference
    if custom_images:
        for filename in tqdm(os.listdir(args.image_dir_path), desc="Inference", leave=False):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            path = os.path.join(args.image_dir_path, filename)
            x = load_image(path).to(device)
            with torch.no_grad():
                y_pred = model(x)
                save_output(y_pred, os.path.join(output_dir, "pred_"+filename))

                tqdm.write(f"Saved predictions to {os.path.join(output_dir, "pred_"+filename)}")
    else:
        idx = 0
        for x, y in tqdm(test_loader, desc="Inference", leave=False):
            x, y = x.to(device), y.to(device)
            cur_file_name = f"buildings_{idx}_real_B.png"
            with torch.no_grad():
                y_pred = model(x)
                save_output(y, os.path.join(output_dir, cur_file_name))
                save_output(y_pred, os.path.join(output_dir, cur_file_name.replace("real", "fake")))
                
                tqdm.write(f"[{idx}] Saved predictions to {os.path.join(output_dir, cur_file_name.replace("real", "fake"))}")

                idx += 1



if __name__ == "__main__":
    inference()


