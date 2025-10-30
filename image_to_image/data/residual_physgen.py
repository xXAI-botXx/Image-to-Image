"""
A PhysGen Dataset Wrapper to get base-propagation and 
complex-propagation in one dataloader.

See:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .physgen import PhysGenDataset



# ---------------------------
#         > Helper <
# ---------------------------
def to_device(dataset):
    """
    Move dataset tensors to the appropriate device (CPU or GPU).

    This helper function expects a dataset item formatted as 
    [input_tensor, target_tensor, index]. It automatically moves 
    all tensor elements to the available device.

    Parameter:
    - dataset (list): 
        A list of three elements: 
        [input_tensor (torch.Tensor), target_tensor (torch.Tensor), index (int)].

    Returns: 
    - list: 
        A list [input_tensor_on_device, target_tensor_on_device, index].

    Raises:
    - ValueError: 
        If the provided dataset item does not have exactly 3 elements.
    """
    # Input: [Tensor(), Tensor(), int]
    if len(dataset) != 3:
        raise ValueError("Expected dataset to be a list of 3 values")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [dataset[0].to(device), dataset[1].to(device), dataset[2]]



# ---------------------------
#        > Dataset <
# ---------------------------
class PhysGenResidualDataset(Dataset):
    """
    Dataset wrapper combining multiple PhysGen dataset variations 
    for residual learning experiments.

    This dataset constructs three related PhysGen datasets:
    1. Baseline dataset (sound_baseline → 'standard' output)
    2. Complex dataset (user-selected variation → 'complex_only' output)
    3. Fusion dataset (user-selected variation → 'standard' output)

    It is designed for residual or multi-source learning setups 
    where the model uses both baseline and complex physics simulations.
    """
    def __init__(self, variation="sound_baseline", mode="train", 
                 fake_rgb_output=False, make_14_dividable_size=False,
                 reflexion_channels=False, reflexion_steps=36, reflexions_as_channels=False):
        """
        Initialize the PhysGenResidualDataset with multiple data sources.

        Parameter:
        - variation (str, default='sound_baseline'): 
            Specifies which physics variation to use for the complex and fusion datasets.
            Common options: {'sound_reflection', 'sound_diffraction', 'sound_combined'}.
        - mode (str, default='train'): 
            Specifies dataset mode. Options: {'train', 'validation'}.
        - fake_rgb_output (bool, default=False): 
            If True, single-channel inputs are expanded to fake RGB format.
        - make_14_dividable_size (bool, default=False): 
            If True, ensures images are resized so that their height and width are divisible by 14.
        - reflexion_channels (bool, default=False): 
            If ray-traces should add to the input. Only for the complex part.
        - reflexion_steps (int, default=36): 
            Defines how many traces should get created.
        - reflexions_as_channels (bool, default=False): 
            If True, every trace gets its own channel, else every trace in one channel.
        """
        self.train_dataset_base = PhysGenDataset(mode='train', variation="sound_baseline", input_type="osm", output_type="standard", 
                                                 fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)
        self.val_dataset_base = PhysGenDataset(mode='validation', variation="sound_baseline", input_type="osm", output_type="standard", 
                                               fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)

        self.train_dataset_complex = PhysGenDataset(mode='train', variation=variation, input_type="osm", output_type="complex_only", 
                                                    fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size, 
                                                    reflexion_channels=reflexion_channels, reflexion_steps=reflexion_steps, reflexions_as_channels=reflexions_as_channels)
        self.val_dataset_complex = PhysGenDataset(mode='validation', variation=variation, input_type="osm", output_type="complex_only", 
                                                  fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size,
                                                  reflexion_channels=reflexion_channels, reflexion_steps=reflexion_steps, reflexions_as_channels=reflexions_as_channels)

        self.train_dataset_fusion = PhysGenDataset(mode='train', variation=variation, input_type="osm", output_type="standard", 
                                                   fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)
        self.val_dataset_fusion = PhysGenDataset(mode='validation', variation=variation, input_type="osm", output_type="standard", 
                                                 fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)
        
        self.datasets = [(self.train_dataset_base, self.val_dataset_base), (self.train_dataset_complex, self.val_dataset_complex), (self.train_dataset_fusion, self.val_dataset_fusion)]

    def __len__(self):
        """
        Return the number of samples in the baseline training dataset.

        Returns:
        - int: Number of samples in the training split of the baseline dataset.
        """
        return len(self.train_dataset_base)

    def __getitem__(self, idx, is_validation=False):
        """
        Retrieve a combined sample across baseline, complex, and fusion datasets.

        This method returns a tuple containing:
        - inputs: (base_input, complex_input)
        - targets: (base_target, complex_target, full_target)

        Parameter:
        - idx (int): 
            Index of the sample to retrieve.
        - is_validation (bool, default=False): 
            If True, samples are drawn from the validation split; 
            otherwise, from the training split.

        Returns:
        - tuple: 
            ((base_input, complex_input), (base_target, complex_target, full_target))
        """
        data_idx = 1 if is_validation else 0

        base_input, base_target = self.datasets[0][data_idx][idx]
        complex_input, complex_target = self.datasets[1][data_idx][idx]
        _, target_ = self.datasets[2][data_idx][idx]

        return (base_input, complex_input), (base_target, complex_target, target_)









