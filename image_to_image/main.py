# main.py
from .utils.argument_parsing import parse_args



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
    else:
        from .model_interactions.inference import run_inference
        run_inference(args)



if __name__ == '__main__':
    main()



