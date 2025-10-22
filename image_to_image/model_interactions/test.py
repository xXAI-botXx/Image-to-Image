# ---------------------------
#        > Imports <
# ---------------------------
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn

from ..utils.argument_parsing import parse_args
from .train import evaluate  # reuse the evaluate function
from ..data.physgen import PhysGenDataset
from ..models.resfcn import ResFCN 


# ---------------------------
#        > Run Test <
# ---------------------------
def test(args=None):
    if args is None:
        args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset loading
    test_dataset = PhysGenDataset(variation=args.data_variation, mode="test", input_type=args.input_type, output_type=args.output_type, 
                                  fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model Loading
    model = ResFCN(in_channels=args.resfcn_in_channels, hidden_channels=args.resfcn_hidden_channels, out_channels=args.resfcn_out_channels, num_blocks=args.resfcn_num_blocks).to(device)

    # Load weights from checkpoint
    checkpoint = torch.load(args.model_params_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # Criterion for evaluation
    if args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"'{args.loss}' is not a supported loss.")

    # Run evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")



if __name__ == "__main__":
    test()


