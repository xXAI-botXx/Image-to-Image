# ---------------------------
#        > Imports <
# ---------------------------
import argparse
import textwrap



# ---------------------------
#     > Argument Parser <
# ---------------------------
def get_arg_parser():
    parser = argparse.ArgumentParser(description="Image-to-Image Framework - train and inferencing")

    # General Parameter
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'inference'],
                        help='Modus: train, test or inference')

    # Trainingsparameter
    parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints', help='Path to save the model checkpoints. Is builded: checkpoint_save_dir/experiment_name/run_name')
    parser.add_argument('--save_only_best_model', action='store_true', help='Should every checkpoint be saved or only the best model?')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Every x epochs checkpoint will be saved (if not `save_only_best_model` is active).')
    parser.add_argument('--validation_interval', type=int, default=5, help='Every x epochs validation will be calculated.')
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

    parser.add_argument('--optimizer', type=str, default="adam", choices=['adam', 'adamw'],
                        help='Optimizer, which decides how exactly to calculate the loss and weight gradients.')
    parser.add_argument('--optimizer_2', type=str, default="adam", choices=['adam', 'adamw'],
                        help='Optimizer, which decides how exactly to calculate the loss and weight gradients -> for the second model(-part).\nFor example for the discriminator part of pix2pix model or the complex part in the residual design model.')
    parser.add_argument('--weight_decay', action="store_true", help='Whether or not to use weight decay (keeping weights smaller).')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='Coefficient of weight decay -> weighting of the penalty.')
    parser.add_argument('--gradient_clipping', action="store_true", help='Whether or not to use gradient clipping.')
    parser.add_argument('--gradient_clipping_threshold', type=float, default=0.5, help='Coefficient of gradient clipping -> threshold for clipping.')
    parser.add_argument('--scheduler', type=str, default="step", choices=['step', 'cosine'],
                        help='Decides how to update the learnrate dynamically.')
    parser.add_argument('--scheduler_2', type=str, default="step", choices=['step', 'cosine'],
                        help='Decides how to update the learnrate dynamically -> for the second model(-part).\nFor example for the discriminator part of pix2pix model or the complex part in the residual design model.')
    parser.add_argument('--use_warm_up', action="store_true", help="Whether to use warm up for optimizer/lr.")
    parser.add_argument('--warm_up_start_lr', type=float, default=0.00005, help='Warm-Up Start learning rate will end at the lr.')
    parser.add_argument('--warm_up_step_duration', type=int, default=1000, help='Duration of increasing learning rate in steps (one step = one batch process).')
    parser.add_argument('--activate_amp', action="store_true", help='Activates (Automatically) Mixed Precision and use scaler to loose no details because of the smaller float.')
    parser.add_argument('--amp_scaler', type=str, default=None, choices=[None, 'grad'],
                        help='Decides whichscaler should be used-')

    # Inference
    parser.add_argument('--model_params_path', type=str, required=False, help='Path to the model checkpoints.')
    parser.add_argument('--image_dir_path', type=str, default=None, required=False, help='Path to folder with images for inference.')
    parser.add_argument('--output_dir', type=str, default='../../data/eval', help='Path to save the real and predicted Images.')

    # Model Loading
    parser.add_argument('--model', type=str, default="resfcn", choices=['resfcn', 'pix2pix', 'residual_design_model', 'physicsformer'],
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
    parser.add_argument('--pix2pix_second_loss_lambda', type=float, default=100.0, help='Weighting of second loss.')

    # ---> PhysicsFormer
    parser.add_argument('--physicsformer_in_channels', type=int, default=1, help='How much channels as input?')
    parser.add_argument('--physicsformer_out_channels', type=int, default=1, help='How much channels as output?')
    parser.add_argument('--physicsformer_img_size', type=int, default=256, help='Size of the image (width or height).')
    parser.add_argument('--physicsformer_patch_size', type=int, default=4, help='Size of patches.')
    parser.add_argument('--physicsformer_embedded_dim', type=int, default=1024, help='Dimension size of embedding.')
    parser.add_argument('--physicsformer_num_blocks', type=int, default=8, help='Number of transformer blocks.')
    parser.add_argument('--physicsformer_heads', type=int, default=16)
    parser.add_argument('--physicsformer_mlp_dim', type=int, default=2048, help='Dimension of MLP.')
    parser.add_argument('--physicsformer_dropout', type=float, default=0.1, help='Dropout rate.')

    # ---> Residual Design Model
    parser.add_argument('--base_model', type=str, default="pix2pix", choices=['resfcn', 'pix2pix'],
                        help='Model to predict base propagation.')
    parser.add_argument('--complex_model', type=str, default="pix2pix", choices=['resfcn', 'pix2pix'],
                        help='Model to predict complex part of propagation (e.g. only reflection).')
    parser.add_argument('--combine_mode', type=str, default="nn", choices=['math', 'nn'],
                        help='Using math calculation or CNN for combining sub predictions.')
    
    parser.add_argument('--loss_2', type=str, default='l1', choices=['l1', 'crossentropy', 'weighted_combined'],
                        help='Loss Function.')
    # ---> WeightedCombinedLoss parameters
    parser.add_argument('--wc_loss_silog_lambda_2', type=float, default=0.5, help='Lambda parameter for SILog loss.')
    parser.add_argument('--wc_loss_weight_silog_2', type=float, default=0.5, help='Weight for SILog loss.')
    parser.add_argument('--wc_loss_weight_grad_2', type=float, default=10.0, help='Weight for gradient loss.')
    parser.add_argument('--wc_loss_weight_ssim_2', type=float, default=5.0, help='Weight for SSIM loss.')
    parser.add_argument('--wc_loss_weight_edge_aware_2', type=float, default=10.0, help='Weight for edge-aware loss.')
    parser.add_argument('--wc_loss_weight_l1_2', type=float, default=1.0, help='Weight for L1 loss.')
    parser.add_argument('--wc_loss_weight_var_2', type=float, default=1.0, help='Weight for variance loss.')
    parser.add_argument('--wc_loss_weight_range_2', type=float, default=1.0, help='Weight for range loss.')
    parser.add_argument('--wc_loss_weight_blur_2', type=float, default=1.0, help='Weight for blur loss.')
    
    # ---> ResFCN Model 2
    parser.add_argument('--resfcn_2_in_channels', type=int, default=1, help='How much channels as input?')
    parser.add_argument('--resfcn_2_hidden_channels', type=int, default=64, help='How much channels in the hidden layers?')
    parser.add_argument('--resfcn_2_out_channels', type=int, default=1, help='How much channels as output?')
    parser.add_argument('--resfcn_2_num_blocks', type=int, default=16, help='How many Residual Blocks should be stacked.')

    # ---> Pix2Pix Model 2
    parser.add_argument('--pix2pix_2_in_channels', type=int, default=1, help='How much channels as input?')
    parser.add_argument('--pix2pix_2_hidden_channels', type=int, default=64, help='How much channels in the hidden layers?')
    parser.add_argument('--pix2pix_2_out_channels', type=int, default=1, help='How much channels as output?')
    parser.add_argument('--pix2pix_2_second_loss_lambda', type=float, default=100.0, help='Weighting of second loss.')

    # ---> PhysicsFormer Model 2
    parser.add_argument('--physicsformer_in_channels_2', type=int, default=1, help='How much channels as input?')
    parser.add_argument('--physicsformer_out_channels_2', type=int, default=1, help='How much channels as output?')
    parser.add_argument('--physicsformer_img_size_2', type=int, default=256, help='Size of the image (width or height).')
    parser.add_argument('--physicsformer_patch_size_2', type=int, default=4, help='Size of patches.')
    parser.add_argument('--physicsformer_embedded_dim_2', type=int, default=1026, help='Dimension size of embedding.')
    parser.add_argument('--physicsformer_num_blocks_2', type=int, default=8, help='Number of transformer blocks.')
    parser.add_argument('--physicsformer_heads_2', type=int, default=16)
    parser.add_argument('--physicsformer_mlp_dim_2', type=int, default=2048, help='Dimension of MLP.')
    parser.add_argument('--physicsformer_dropout_2', type=float, default=0.1, help='Dropout rate.')


    # Data
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
    parser.add_argument('--run_name', type=str, default="image-to-image", help='Name of the specific run. Will be used for naming but will add "YEAR-MONTH-DAY_HOUR_MINUTE" in front of your choosen name.')
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


