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
    # Input: [Tensor(), Tensor(), int]
    if len(dataset) != 3:
        raise ValueError("Expected dataset to be a list of 3 values")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [dataset[0].to(device), dataset[1].to(device), dataset[2]]



# ---------------------------
#        > Dataset <
# ---------------------------
class PhysGenResidualDataset(Dataset):

    def __init__(self, variation="sound_baseline", mode="train", 
                 fake_rgb_output=False, make_14_dividable_size=False):
        self.train_dataset_base = PhysGenDataset(mode='train', variation="sound_baseline", input_type="osm", output_type="standard", fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)
        self.val_dataset_base = PhysGenDataset(mode='validation', variation="sound_baseline", input_type="osm", output_type="standard", fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)

        self.train_dataset_complex = PhysGenDataset(mode='train', variation=variation, input_type="osm", output_type="complex_only", fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)
        self.val_dataset_complex = PhysGenDataset(mode='validation', variation=variation, input_type="osm", output_type="complex_only", fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)

        self.train_dataset_fusion = PhysGenDataset(mode='train', variation=variation, input_type="osm", output_type="standard", fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)
        self.val_dataset_fusion = PhysGenDataset(mode='validation', variation=variation, input_type="osm", output_type="standard", fake_rgb_output=fake_rgb_output, make_14_dividable_size=make_14_dividable_size)
        
        self.datasets = [(self.train_dataset_base, self.val_dataset_base), (self.train_dataset_complex, self.val_dataset_complex), (self.train_dataset_fusion, self.val_dataset_fusion)]

    def __len__(self):
        return len(self.train_dataset_base)

    def __getitem__(self, idx, is_validation=False):
        data_idx = 1 if is_validation else 0

        base_input, base_target, _ = self.datasets[0][data_idx][idx]
        complex_input, complex_target, _ = self.datasets[1][data_idx][idx]
        _, target_, idx_ = self.datasets[2][data_idx][idx]

        return (base_input, complex_input), (base_target, complex_target, target_)









