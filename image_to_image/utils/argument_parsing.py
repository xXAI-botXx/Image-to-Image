import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Image-to-Image Framework")

    # General Parameter
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'inference', 'physgen_benchmark'],
                        help='Modus: train, test oder inference')

    # Trainingsparameter
    parser.add_argument('--epochs', type=int, default=50, help='Amount of whole data loops.')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of a batch, data is processed in batches (smaller packages) and the GPU processes then one batch at a time.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learnrate of adjusting the weights towards the gradients.')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1'],
                        help='Loss Function.')
    parser.add_argument('--optimizer', type=str, default="adam", choices=['adam'],
                        help='Optimizer, which decides how exactly to calculate the loss and weight gradients.')
    parser.add_argument('--scheduler', type=str, default="step", choices=['step'],
                        help='Decides how to update the learnrate dynamically.')
    # extra arguments for scheduler? (for parameters)
    parser.add_argument('--scaler', type=str, default=None, choices=[None, 'grad'],
                        help='Scaling of loss.')
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Path to save the model checkpoints.')

    # Inference
    parser.add_argument('--model_params_path', type=str, required=False, help='Path to the model checkpoints.')
    parser.add_argument('--output_dir', type=str, default='../../data/eval', help='Path to save the real and predicted Images.')

    # Model Loading
    parser.add_argument('--model', type=str, default="resfcn", choices=['resfcn'],
                        help='Which Model should be choosen')

    # ---> ResFCN
    parser.add_argument('--resfcn_in_channels', type=int, default=1, help='How much channels as input?')
    parser.add_argument('--resfcn_hidden_channels', type=int, default=64, help='How much channels in the hidden layers?')
    parser.add_argument('--resfcn_out_channels', type=int, default=1, help='How much channels as output?')
    parser.add_argument('--resfcn_num_blocks', type=int, default=16, help='How many Residual Blocks should be stacked.')

    # Data
    # parser.add_argument('--data_mode', type=str, default='train',
    #                     choices=['train', 'test', 'eval'],
    #                     help='Which data version should be loaded')
    parser.add_argument('--data_variation', type=str, default='sound_baseline', choices=['sound_baseline', 'sound_reflection', 'sound_diffraction', 'sound_combined'],
                        help='Name of the dataset variation.')
    parser.add_argument('--input_type', type=str, default='osm', choices=['osm', 'base_simulation'],
                        help='Input type (can be used to get the base simulation/propagation as input).')
    parser.add_argument('--output_type', type=str, default='standard', choices=['standard', 'complex_only'],
                        help='Output Type (can be used to get only reflexion as target).')
    parser.add_argument('--fake_rgb_output', action='store_true',
                        help='If setted: Input image is putted with 3 channels.')
    parser.add_argument('--make_14_dividable_size', action='store_true',
                        help='Adjusts imagesizes to a multiple of 14 if setted (needed for some networks).')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Rechen-Device')
    
    # Experiment Tracking
    parser.add_argument('--experiment_name', type=str, default="image-to-image", help='Name of the overall experiment (will stay the same over most runs).')
    parser.add_argument('--run_name', type=str, default="image-to-image", help='Name of the specific run.')
    parser.add_argument('--tensorboard_path', type=str, default="../tensorboard", help='Where should the results from tensorboard be saved to?')
    parser.add_argument('--save_path', type=str, default="../train_inference", help='Where should the results from your model be saved to?')
    parser.add_argument('--cmap', type=str, default="gray", help='Color Map for saving images.')


    return parser



def parse_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    return args


