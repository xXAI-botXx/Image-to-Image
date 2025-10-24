# ---------------------------
#        > Imports <
# ---------------------------
import sys
sys.path += ["."]

import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning)

import image_to_image as iti
from image_to_image.utils import parse_args
from image_to_image.model_interactions import train, inference, test



# ---------------------------
#         > Main <
# ---------------------------
def main():
    args = parse_args()

    print(f"Running in mode: {args.mode}")
    print(f"Using device: {args.device}")

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'inference':
        inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")



if __name__ == '__main__':
    main()



