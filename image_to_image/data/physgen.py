"""
PhysGen Dataset Loader

PyTorch DataLoader.

Also provide some functions for downloading the dataset.

See:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""
# ---------------------------
#        > Imports <
# ---------------------------
import os
import shutil
from PIL import Image

from datasets import load_dataset

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
from torchvision import transforms

import img_phy_sim as ips
import prime_printer as prime



# ---------------------------
#        > Helper <
# ---------------------------
def resize_tensor_to_divisible_by_14(tensor: torch.Tensor) -> torch.Tensor:
    """
    Resize a tensor so that its height and width are divisible by 14.

    This function ensures the spatial dimensions (H, W) of a given tensor 
    are compatible with architectures that require sizes divisible by 14 
    (e.g., ResNet, ResFCN). It resizes using bilinear interpolation.

    Parameter:
    - tensor (torch.Tensor): 
        Input tensor of shape (C, H, W) or (B, C, H, W).

    Returns:
    - torch.Tensor: 
        Resized tensor with dimensions divisible by 14.

    Raises:
    - ValueError: 
        If the tensor has neither 3 nor 4 dimensions.
    """
    if tensor.dim() == 3:
        c, h, w = tensor.shape
        new_h = h - (h % 14)
        new_w = w - (w % 14)
        return F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    
    elif tensor.dim() == 4:
        b, c, h, w = tensor.shape
        new_h = h - (h % 14)
        new_w = w - (w % 14)
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    else:
        raise ValueError("Tensor must be 3D (C, H, W) or 4D (B, C, H, W)")



# ---------------------------
#        > Dataset <
# ---------------------------
class PhysGenDataset(Dataset):
    """
    PyTorch Dataset wrapper for the PhysicsGen dataset.

    Loads the PhysGen dataset from Hugging Face and provides configurable
    input/output modes for physics-based generative learning tasks.

    The dataset contains Open Sound Maps (OSM) and simulated soundmaps
    for tasks involving sound propagation modeling.
    """
    def __init__(self, variation="sound_baseline", mode="train", input_type="osm", output_type="standard", 
                 fake_rgb_output=False, make_14_dividable_size=False,
                 reflexion_channels=False, reflexion_steps=36, reflexions_as_channels=False):
        """
        Loads PhysGen Dataset.

        Parameter:
        - variation (str, default='sound_baseline'): 
            Dataset variation to load. Options include:
            {'sound_baseline', 'sound_reflection', 'sound_diffraction', 'sound_combined'}.
        - mode (str, default='train'): 
            Dataset split to use. Options: {'train', 'test', 'validation'}.
        - input_type (str, default='osm'): 
            Defines the input image type:
            - 'osm': open sound map input.
            - 'base_simulation': uses the baseline sound simulation as input.
        - output_type (str, default='standard'): 
            Defines the output image type:
            - 'standard': full soundmap prediction.
            - 'complex_only': difference from baseline soundmap.
        - fake_rgb_output (bool, default=False): 
            If True, replicates single-channel inputs to fake RGB (3-channel).
        - make_14_dividable_size (bool, default=False): 
            If True, resizes tensors so that height and width are divisible by 14.
        - reflexion_channels (bool, default=False): 
            If ray-traces should add to the input.
        - reflexion_steps (int, default=36): 
            Defines how many traces should get created.
        - reflexions_as_channels (bool, default=False): 
            If True, every trace gets its own channel, else every trace in one channel.
        """
        self.fake_rgb_output = fake_rgb_output
        self.make_14_dividable_size = make_14_dividable_size
        self.reflexion_channels = reflexion_channels
        self.reflexion_steps = reflexion_steps
        self.reflexions_as_channels = reflexions_as_channels

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # get data
        self.dataset = load_dataset("mspitzna/physicsgen", name=variation, trust_remote_code=True)
        # print("Keys:", self.dataset.keys())
        self.dataset = self.dataset[mode]
        
        self.input_type = input_type
        self.output_type = output_type
        if self.input_type == "base_simulation" or self.output_type == "complex_only":
            self.basesimulation_dataset = load_dataset("mspitzna/physicsgen", name="sound_baseline", trust_remote_code=True)
            self.basesimulation_dataset = self.basesimulation_dataset[mode]

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] PIL image to [0,1] FloatTensor
        ])
        print(f"PhysGen ({variation}) Dataset for {mode} got created")

    def __len__(self):
        """
        Returns the number of available samples.

        Returns:
        - int: 
            Number of samples in the dataset split.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve an input-target pair from the dataset.

        This function loads the input image and corresponding target image,
        applies transformations (resizing, fake RGB, etc.), and returns
        them as PyTorch tensors.

        Parameter: 
        - idx (int): 
            Index of the data sample.

        Returns: 
        - tuple[torch.Tensor, torch.Tensor]<br>
            A tuple containing:
            - input_img : torch.Tensor<br>
                Input tensor (shape: [C, H, W]).
            - target_img : torch.Tensor<br>
                Target tensor (shape: [C, H, W]).
        """
        sample = self.dataset[idx]
        # print(sample)
        # print(sample.keys())
        if self.input_type == "base_simulation":
            input_img = self.basesimulation_dataset[idx]["soundmap"]
        else:
            input_img = sample["osm"]  # PIL Image
        target_img = sample["soundmap"]  # PIL Image

        input_img = self.transform(input_img)
        target_img = self.transform(target_img)

        # Fix real image size 512x512 > 256x256
        input_img = F.interpolate(input_img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        input_img = input_img.squeeze(0)
        # target_img = target_img.unsqueeze(0)

        # change size
        if self.make_14_dividable_size:
            input_img = resize_tensor_to_divisible_by_14(input_img)
            target_img = resize_tensor_to_divisible_by_14(target_img)

        # add fake rgb
        if self.fake_rgb_output and input_img.shape[0] == 1:  # shape (1, H, W)
            input_img = input_img.repeat(3, 1, 1)  # make it (B, 3, H, W)

        if self.output_type == "complex_only":
            base_simulation_img = self.transform(self.basesimulation_dataset[idx]["soundmap"])
            # base_simulation_img = resize_tensor_to_divisible_by_14(self.transform(self.basesimulation_dataset[idx]["soundmap"]))
            # target_img = torch.abs(target_img[0] - base_simulation_img[0])
            target_img = target_img[0] - base_simulation_img[0]
            target_img = target_img.unsqueeze(0)
            target_img *= -1

        # add raytracing
        if self.reflexion_channels:
            rays = ips.ray_tracing.trace_beams(rel_position=(0.5, 0.5),	
                                                img_src=np.squeeze(input_img.cpu().numpy(), axis=0),	
                                                directions_in_degree=ips.math.get_linear_degree_range(step_size=(self.reflexion_steps/360)*100),	
                                                wall_values=[0],	
                                                wall_thickness=0,	
                                                img_border_also_collide=False,	
                                                reflexion_order=3,	
                                                should_scale_rays=True,	
                                                should_scale_img=False)
            ray_img = ips.ray_tracing.draw_rays(rays,	
                                                detail_draw=False,	
                                                output_format='channels' if self.reflexions_as_channels else 'single_image',	
                                                img_background=None,	
                                                ray_value=[50, 100, 255],	
                                                ray_thickness=1,	
                                                img_shape=(256, 256),
                                                should_scale_rays_to_image=True,
                                                show_only_reflections=True)
            # (256, 256)
            # print("CHECKPOINT")
            # print(ray_img.shape)
            ray_img = self.transform(ray_img)
            ray_img = ray_img.float()
            if ray_img.ndim == 2:
                ray_img = ray_img.unsqueeze(0)  # (1, H, W)

            # print(ray_img.shape)
            # print(input_img.shape)
            # Merging with input image 
            if ray_img.shape[1:] == input_img.shape[1:]:
                input_img = torch.cat((input_img, ray_img), dim=0)
            else:
                raise ValueError(f"Ray image shape {ray_img.shape} does not match input image shape {input_img.shape}.")

        return input_img, target_img



# ---------------------------
#   > Helpful Functions <
# ---------------------------
# For external not internal

def get_dataloader(mode='train', variation="sound_reflection", input_type="osm", output_type="complex_only", shuffle=True):
    """
    Create a PyTorch DataLoader for the PhysGen dataset.

    This helper simplifies loading the PhysGen dataset for training,
    validation, or testing.

    Parameter:
    - mode (str, default='train'): 
        Dataset split to use ('train', 'test', 'validation').
    - variation (str, default='sound_reflection'): 
        Dataset variation to load.
    - input_type (str, default='osm'): 
        Defines the input type ('osm' or 'base_simulation').
    - output_type (str, default='complex_only'): 
        Defines the output type ('standard' or 'complex_only').
    - shuffle (bool, default=True): 
        Whether to shuffle the dataset between epochs.

    Returns:
    - torch.utils.data.DataLoader: 
        DataLoader that provides batches of PhysGen samples.
    """
    dataset = PhysGenDataset(mode=mode, variation=variation, input_type=input_type, output_type=output_type)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=1)



def get_image(mode='train', variation="sound_reflection", input_type="osm", output_type="complex_only", shuffle=True, 
              return_output=False, as_numpy_array=True):
    """
    Retrieve one image (input and optionally output) from the PhysGen dataset.

    Provides an easy way to visualize or inspect a single PhysGen sample
    without manually instantiating a DataLoader.

    Parameter:
    - mode (str, default='train'): 
        Dataset split ('train', 'test', 'validation').
    variation (str, default='sound_reflection'): 
        Dataset variation.
    input_type (str, default='osm'): 
        Defines the input type ('osm' or 'base_simulation').
    output_type (str, default='complex_only'): 
        Defines the output type ('standard' or 'complex_only').
    shuffle (bool, default=True): 
        Randomly select the sample.
    return_output (bool, default=False): 
        If True, returns both input and target tensors.
    as_numpy_array (bool, default=True): 
        If True, converts tensors to NumPy arrays for easier visualization.

    Returns: 
    - numpy.ndarray or list[numpy.ndarray]: 
        Input image as NumPy array, or a list [input, target] if `return_output` is True.
    """
    dataset = PhysGenDataset(mode=mode, variation=variation, input_type=input_type, output_type=output_type)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=1)
    cur_data = next(iter(loader))
    input_ = cur_data[0]
    output_ = cur_data[1]

    if as_numpy_array:
        input_ = input_.detach().cpu().numpy()
        output_ = output_.detach().cpu().numpy()

        # remove batch channel
        input_ = np.squeeze(input_, axis=0)
        output_ = np.squeeze(output_, axis=0)

        if len(input_.shape) == 3:
            input_ = np.squeeze(input_, axis=0)
            output_ = np.squeeze(output_, axis=0)

        input_ = np.transpose(input_, (1, 0))
        output_ = np.transpose(output_, (1, 0))


    result = input_
    if return_output:
        result = [input_, output_]

    return result



def save_dataset(output_real_path, output_osm_path, 
                 variation, input_type, output_type,
                 data_mode, 
                 info_print=False, progress_print=True):
    """
    Save PhysGen dataset samples as images to disk.

    This function loads the specified PhysGen dataset, converts input and
    target tensors to images, and saves them as `.png` files for inspection,
    debugging, or model-agnostic data use.

    Parameter: 
    - output_real_path (str): 
        Directory to save target (real) soundmaps.
    - output_osm_path (str): 
        Directory to save input (OSM) maps.
    - variation (str): 
        Dataset variation (e.g. 'sound_reflection').
    - input_type (str): 
        Input type ('osm' or 'base_simulation').
    - output_type (str): 
        Output type ('standard' or 'complex_only').
    - data_mode (str): 
        Dataset split ('train', 'test', 'validation').
    - info_print (bool, default=False): 
        If True, prints detailed information for each saved sample.
    - progress_print (bool, default=True): 
        If True, shows progress updates in the console.

    Raises:
    - ValueError: 
        If image data falls outside the valid range [0, 255].

    """
    # Clearing
    if os.path.exists(output_osm_path) and os.path.isdir(output_osm_path):
        shutil.rmtree(output_osm_path)
        os.makedirs(output_osm_path)
        print(f"Cleared {output_osm_path}.")
    else:
        os.makedirs(output_osm_path)
        print(f"Created {output_osm_path}.")

    if os.path.exists(output_real_path) and os.path.isdir(output_real_path):
        shutil.rmtree(output_real_path)
        os.makedirs(output_real_path)
        print(f"Cleared {output_real_path}.")
    else:
        os.makedirs(output_real_path)
        print(f"Created {output_real_path}.")
    
    # Load Dataset
    dataset = PhysGenDataset(mode=data_mode, variation=variation, input_type=input_type, output_type=output_type)
    data_len = len(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Save Dataset
    for i, data in enumerate(dataloader):
        if progress_print:
            prime.get_progress_bar(total=data_len, progress=i+1, 
                                   should_clear=True, left_bar_char='|', right_bar_char='|', 
                                   progress_char='#', empty_char=' ', 
                                   front_message='Physgen Data Loading', back_message='', size=15)

        input_img, target_img, idx = data
        idx = idx[0].item() if isinstance(idx, torch.Tensor) else idx

        if info_print:
            print(f"Prediction shape [osm]: {input_img.shape}")
            print(f"Prediction shape [target]: {target_img.shape}")

            print(f"OSM Info:\n    -> shape: {input_img.shape}\n    -> min: {input_img.min()}, max: {input_img.max()}")

        real_img = target_img.squeeze(0).cpu().squeeze(0).detach().numpy()
        if not (0 <= real_img.min() <= 255 and 0 <= real_img.max() <=255):
            raise ValueError(f"Real target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")
        if info_print:
            print( f"\nReal target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")
        if real_img.max() <= 1.0:
            real_img *= 255
        if info_print:
            print( f"Real target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")
        real_img = real_img.astype(np.uint8)
        if info_print:
            print( f"Real target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")

        if len(input_img.shape) == 4:
            osm_img = input_img[0, 0].cpu().detach().numpy()
        else:
            osm_img = input_img[0].cpu().detach().numpy()
        if not (0 <= osm_img.min() <= 255 and 0 <= osm_img.max() <=255):
            raise ValueError(f"Real target has values out of 0-256 range => min:{osm_img.min()}, max:{osm_img.max()}")
        if osm_img.max() <= 1.0:
            osm_img *= 255
        osm_img = osm_img.astype(np.uint8)

        if info_print:
            print(f"OSM Info:\n    -> shape: {osm_img.shape}\n    -> min: {osm_img.min()}, max: {osm_img.max()}")

        # Save Results
        file_name = f"physgen_{idx}.png"

        # save pred image
        # save_img = os.path.join(output_pred_path, file_name)
        # cv2.imwrite(save_img, pred_img)
        # print(f"    -> saved pred at {save_img}")

        # save real image
        save_img = os.path.join(output_real_path, "target_"+file_name)
        cv2.imwrite(save_img, real_img)
        if info_print:
            print(f"    -> saved real at {save_img}")

        # save osm image
        save_img = os.path.join(output_osm_path, "input_"+file_name)
        cv2.imwrite(save_img, osm_img)
        if info_print:
            print(f"    -> saved osm at {save_img}")
    print(f"\nSuccessfull saved {data_len} datapoints into {os.path.abspath(output_real_path)} & {os.path.abspath(output_osm_path)}")



# ---------------------------
#     > Dataset Saving <
# ---------------------------
if __name__ == "__main__":
    """
    Command-line interface for saving PhysGen dataset samples.

    Allows users to export the PhysGen dataset as image pairs for a given
    variation, input/output configuration, and mode.

    Example
    -------
    >>> python physgen_dataset_loader.py \
            --output_real_path ./real \
            --output_osm_path ./osm \
            --variation sound_reflection \
            --input_type osm \
            --output_type standard \
            --data_mode train
    """
    import argparse

    parser = argparse.ArgumentParser(description="Save OSM and real PhysGen dataset images.")

    parser.add_argument("--output_real_path", type=str, required=True, help="Path to save real target images")
    parser.add_argument("--output_osm_path", type=str, required=True, help="Path to save OSM input images")
    parser.add_argument("--variation", type=str, required=True, help="PhysGen variation (e.g. box_texture, box_position, etc.)")
    parser.add_argument("--input_type", type=str, required=True, help="Input type (e.g. osm_depth)")
    parser.add_argument("--output_type", type=str, required=True, help="Output type (e.g. real_depth)")
    parser.add_argument("--data_mode", type=str, required=True, help="Data Mode: train, test, val")
    parser.add_argument("--info_print", action="store_true", help="Print additional info")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress printing")

    args = parser.parse_args()

    save_dataset(
        output_real_path=args.output_real_path,
        output_osm_path=args.output_osm_path,
        variation=args.variation,
        input_type=args.input_type,
        output_type=args.output_type,
        data_mode=args.data_mode,
        info_print=args.info_print,
        progress_print=not args.no_progress
    )

    