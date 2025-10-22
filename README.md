# Image-to-Image
An Image-to-Image PyTorch repo. Primary done for PhysGen/Urban Noise Dataset.

Data References:
- [Dataset](https://arxiv.org/abs/2403.10904v2)
- [Main Paper / comparison](https://arxiv.org/abs/2503.05333)
- [Normalizing Flow Approach](https://www.arxiv.org/pdf/2510.04510)


Content:
- [Installation](#installation)
- [Example Commands](#example-commands)
- [Experiment Commands](#experiment-commands)
- [Access Experiment Tracking](#access-experiment-tracking)



> For the Physgen benchmark, you have to first train your model, then open the [Evaluation Physgen-Benchmark Notebook](./image_to_image/model_interactions/eval_physgen_benchmark.ipynb) and run the commands (and adjust the parameters there). 


<br>
<br>

---
### Installation

```bash
conda create -n img-to-img python=3.13 pip -y
conda activate img-to-img
pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Use your CUDA needed. You may want to check with `nvidia-smi`.

<br>
<br>

---
### Example Commands

**Training run**
```bash
python -m main \
  --mode train \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --loss l1 \
  --optimizer adam \
  --scheduler step \
  --scaler grad \
  --save_dir ./checkpoints \
  --model resfcn \
  --resfcn_in_channels 1 \
  --resfcn_hidden_channels 64 \
  --resfcn_out_channels 1 \
  --resfcn_num_blocks 16 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --device cuda \
  --experiment_name image-to-image \
  --run_name resfcn_test \
  --tensorboard_path ./tensorboard \
  --save_path ./mlflow_images \
  --cmap gray
```


**Testing**
```bash
python -m main \
  --mode test \
  --batch_size 16 \
  --loss l1 \
  --model resfcn \
  --model_params_path ./checkpoints/my_model.pth \
  --resfcn_in_channels 1 \
  --resfcn_hidden_channels 64 \
  --resfcn_out_channels 1 \
  --resfcn_num_blocks 16 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --device cuda
```


**Inference**
```bash
python -m main \
  --mode inference \
  --batch_size 16 \
  --model resfcn \
  --model_params_path ./checkpoints/my_model.pth \
  --resfcn_in_channels 1 \
  --resfcn_hidden_channels 64 \
  --resfcn_out_channels 1 \
  --resfcn_num_blocks 16 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --device cuda
```

Or custom data inference:

```bash
python -m main \
  --mode inference \
  --batch_size 16 \
  --model resfcn \
  --model_params_path ./checkpoints/my_model.pth \
  --resfcn_in_channels 1 \
  --resfcn_hidden_channels 64 \
  --resfcn_out_channels 1 \
  --resfcn_num_blocks 16 \
  --image_dir_path ./data/dataset \
  --device cuda
```


> In windows you have to use ``` for the line breaks or remove the line breaks completly. Or you use Linux via Docker.

> You can also use: `python -m main --help` for help.

<br>
<br>

---
### Experiment Commands

Pix2Pix Rebuild:
```bash
python -m main \
  --mode train \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --loss weighted_combined \
  --wc_loss_silog_lambda 0.5 \
  --wc_loss_weight_silog 1.0 \
  --wc_loss_weight_grad 50.0 \
  --wc_loss_weight_ssim 100.0 \
  --wc_loss_weight_edge_aware 50.0 \
  --wc_loss_weight_l1 10.0 \
  --wc_loss_weight_var 0.0 \
  --wc_loss_weight_range 0.0 \
  --wc_loss_weight_blur 0.0 \
  --optimizer adam \
  --scheduler step \
  --scaler grad \
  --save_dir ./checkpoints \
  --model pix2pix \
  --pix2pix_in_channels 1 \
  --pix2pix_hidden_channels 64 \
  --pix2pix_out_channels 1 \
  --pix2pix_second_loss_lambda 100 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --device cuda \
  --experiment_name image-to-image \
  --run_name pix2pix_rebuild_test \
  --tensorboard_path ./tensorboard \
  --save_path ./mlflow_images \
  --cmap gray
```

New Simple Model try out:
```bash
python -m main \
  --mode train \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --loss weighted_combined \
  --wc_loss_silog_lambda 0.5 \
  --wc_loss_weight_silog 1.0 \
  --wc_loss_weight_grad 50.0 \
  --wc_loss_weight_ssim 100.0 \
  --wc_loss_weight_edge_aware 50.0 \
  --wc_loss_weight_l1 10.0 \
  --wc_loss_weight_var 0.0 \
  --wc_loss_weight_range 0.0 \
  --wc_loss_weight_blur 0.0 \
  --optimizer adam \
  --scheduler step \
  --scaler grad \
  --save_dir ./checkpoints \
  --model resfcn \
  --resfcn_in_channels 1 \
  --resfcn_hidden_channels 64 \
  --resfcn_out_channels 1 \
  --resfcn_num_blocks 16 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --device cuda \
  --experiment_name image-to-image \
  --run_name resfcn_weighted_combined_test \
  --tensorboard_path ./tensorboard \
  --save_path ./mlflow_images \
  --cmap gray
```



<br>
<br>

---
### Access Experiment Tracking

In order to see the history of previous trainings you can start tensorboard of mlflow. Both start a local server which can be acces over an address which is very handy because they often are stored on another device and so can be viewed over SSH.<br>
While tensorboard shows curves for the loss (there are some other features, but this is the most important), does mlflow shows different metrices, parameters and model-artifacts (and also optionally some inferences).

You have to start one of them:
1. Go with a terminal to the top folder:
```bash
mlflow ui
```

Or with tensorboard:
```bash
tensorboard --logdir your_save_dir/tensorboard
```

Both starts a local website:
- mlflow most likely: http://localhost:5000
- tensorboard most likely: http://localhost:6006



