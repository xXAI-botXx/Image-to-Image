# ---------------------------
#        > Imports <
# ---------------------------
# import sys
# import os

# # Ensure relative imports work correctly when run directly
# if __package__ is None or __package__ == '':
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .utils.argument_parsing import parse_args



# ---------------------------
#         > Main <
# ---------------------------
def main():
    args = parse_args()

    print(f"Running in mode: {args.mode}")
    print(f"Using device: {args.device}")

    if args.mode == 'train':
        from .model_interactions.train import train
        train(args)
    elif args.mode == 'test':
        from .model_interactions.test import test
        test(args)
    elif args.mode == 'inference':
        from model_interactions.inference import inference
        inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")



if __name__ == '__main__':
    main()



