# ---------------------------
#        > Imports <
# ---------------------------
import os

import cv2

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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension


def save_output(tensor, path):
    from torchvision.utils import save_image
    save_image(tensor, path)


# ---------------------------
#       > Inference <
# ---------------------------
def inference(args=None):
    if args is None:
        args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data Loading
    test_dataset = PhysGenDataset(variation=args.data_variation, mode="test", input_type=args.input_type, output_type=args.output_type, 
                                  fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Model Loading
    model = ResFCN(in_channels=1, hidden_channels=128, out_channels=1, num_blocks=16).to(device)

    checkpoint = torch.load(args.model_params_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # run inference
    # for filename in os.listdir(input_dir):
    #     if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
    #         continue
    #     path = os.path.join(input_dir, filename)
    #     x = load_image(path).to(device)
    idx = 0
    for x, y in test_loader:
        cur_file_name = f"buildings_{idx}_real_B.png"
        with torch.no_grad():
            y_pred = model(x)
            save_output(y, os.path.join(output_dir, cur_file_name))
            save_output(y_pred, os.path.join(output_dir, cur_file_name.replace("real", "fake")))
            idx += 1



if __name__ == "__main__":
    inference()


