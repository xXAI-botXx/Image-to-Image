# ---------------------------
#        > Imports <
# ---------------------------
import argparse
import textwrap



# ---------------------------
#     > Argument Parser <
# ---------------------------
def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Image-to-Image Framework - train and inferencing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Example commands:

        Training:
          python -m main \\
            --mode train \\
            --epochs 100 \\
            --batch_size 16 \\
            --lr 0.0001 \\
            --loss l1 \\
            --optimizer adam \\
            --scheduler step \\
            --scaler grad \\
            --save_dir ./checkpoints \\
            --model resfcn \\
            --resfcn_in_channels 1 \\
            --resfcn_hidden_channels 64 \\
            --resfcn_out_channels 1 \\
            --resfcn_num_blocks 16 \\
            --data_variation sound_reflection \\
            --input_type osm \\
            --output_type standard \\
            --device cuda \\
            --experiment_name image-to-image \\
            --run_name resfcn_test \\
            --tensorboard_path ./tensorboard \\
            --save_path ./mlflow_images \\
            --cmap gray

        Testing:
          python -m main \\
            --mode test \\
            --batch_size 16 \\
            --loss l1 \\
            --model resfcn \\
            --model_params_path ./checkpoints/my_model.pth \\
            --resfcn_in_channels 1 \\
            --resfcn_hidden_channels 64 \\
            --resfcn_out_channels 1 \\
            --resfcn_num_blocks 16 \\
            --data_variation sound_reflection \\
            --input_type osm \\
            --output_type standard \\
            --device cuda

        Inference:
          python -m main \\
            --mode inference \\
            --batch_size 16 \\
            --model resfcn \\
            --model_params_path ./checkpoints/my_model.pth \\
            --resfcn_in_channels 1 \\
            --resfcn_hidden_channels 64 \\
            --resfcn_out_channels 1 \\
            --resfcn_num_blocks 16 \\
            --input_dir_path ./data/dataset \\
            --device cuda
        """)
        )

    # General Parameter
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'inference'],
                        help='Modus: train, test or inference')

    # Trainingsparameter
    parser.add_argument('--epochs', type=int, default=50, help='Amount of whole data loops.')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of a batch, data is processed in batches (smaller packages) and the GPU processes then one batch at a time.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learnrate of adjusting the weights towards the gradients.')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'crossentropy', 'weighted_combined'],
                        help='Loss Function.')
    # ---> WeightedCombinedLoss parameters
    parser.add_argument('--wc_loss_silog_lambda', type=float, default=0.5, help='Lambda parameter for SILog loss.')
    parser.add_argument('--wc_loss_weight_silog', type=float, default=0.5, help='Weight for SILog loss.')
    parser.add_argument('--wc_loss_weight_grad', type=float, default=10.0, help='Weight for gradient loss.')
    parser.add_argument('--wc_loss_weight_ssim', type=float, default=5.0, help='Weight for SSIM loss.')
    parser.add_argument('--wc_loss_weight_edge_aware', type=float, default=10.0, help='Weight for edge-aware loss.')
    parser.add_argument('--wc_loss_weight_l1', type=float, default=1.0, help='Weight for L1 loss.')
    parser.add_argument('--wc_loss_weight_var', type=float, default=1.0, help='Weight for variance loss.')
    parser.add_argument('--wc_loss_weight_range', type=float, default=1.0, help='Weight for range loss.')
    parser.add_argument('--wc_loss_weight_blur', type=float, default=1.0, help='Weight for blur loss.')

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
    parser.add_argument('--image_dir_path', type=str, default=None, required=False, help='Path to folder with images for inference.')
    parser.add_argument('--output_dir', type=str, default='../../data/eval', help='Path to save the real and predicted Images.')

    # Model Loading
    parser.add_argument('--model', type=str, default="resfcn", choices=['resfcn', 'pix2pix'],
                        help='Which Model should be choosen')

    # ---> ResFCN
    parser.add_argument('--resfcn_in_channels', type=int, default=1, help='How much channels as input?')
    parser.add_argument('--resfcn_hidden_channels', type=int, default=64, help='How much channels in the hidden layers?')
    parser.add_argument('--resfcn_out_channels', type=int, default=1, help='How much channels as output?')
    parser.add_argument('--resfcn_num_blocks', type=int, default=16, help='How many Residual Blocks should be stacked.')

    # ---> Pix2Pix
    parser.add_argument('--pix2pix_in_channels', type=int, default=1, help='How much channels as input?')
    parser.add_argument('--pix2pix_hidden_channels', type=int, default=64, help='How much channels in the hidden layers?')
    parser.add_argument('--pix2pix_out_channels', type=int, default=1, help='How much channels as output?')
    parser.add_argument('--pix2pix_second_loss_lambda', type=int, default=100, help='Weighting of second loss.')

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



# ---------------------------
#     > Get Arguments <
# ---------------------------
def parse_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    return args


