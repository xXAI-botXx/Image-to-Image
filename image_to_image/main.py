"""
Module to run image-to-image.<br>
Train, test or inference models.

Functions:
- main

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=UserWarning)

from .utils import parse_args
# from .model_interactions import train, inference, test



# ---------------------------
#         > Main <
# ---------------------------
def main():
    """
    Main entry point for image-to-image tasks.

    This function parses command-line arguments and dispatches
    the workflow based on the specified mode:
        - 'train': runs the training routine
        - 'test': runs evaluation on the test dataset
        - 'inference': runs inference on images or datasets

    Raises:
    - ValueError: if the provided mode is unknown.
    """
    from .model_interactions import train, inference, test

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



