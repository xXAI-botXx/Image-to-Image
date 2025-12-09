# Image-to-Image
An Image-to-Image PyTorch repo. Primary done for PhysGen/Urban Noise Dataset.

Data References:
- [Dataset](https://arxiv.org/abs/2403.10904v2)
- [Main Paper / comparison](https://arxiv.org/abs/2503.05333)
- [Normalizing Flow Approach](https://www.arxiv.org/pdf/2510.04510)


Content:
- [Installation](#installation)
- [Example Bash Output](#example-bash-output)
- [Process](#process)
- [Example Commands](#example-commands)
- [Experiment Commands](#experiment-commands)
- [Access Experiment Tracking](#access-experiment-tracking)



> For the Physgen benchmark, you have to first train your model, then open the [Evaluation Physgen-Benchmark Notebook](./image_to_image/model_interactions/eval_physgen_benchmark.ipynb) and run the commands (and adjust the parameters there). 


[> Documentation <](https://xxai-botxx.github.io/Image-to-Image/image_to_image.html)


<br>
<br>

---
### Installation

Clone/Download the repo and use it then:

  ```bash
  conda create -n img-to-img python=3.13 pip -y
  conda activate img-to-img
  pip install --no-cache-dir -r requirements.txt
  ```
  For a GPU version you can run this afterwards:
  ```bash
  pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```

> Use your CUDA version needed. You may want to check with `nvidia-smi`.


```bash
git clone https://github.com/xXAI-botXx/Physics-U-ViT.git physics_u_vit
```


<br>
<br>

---
### Example Bash Output

```text
attention mode is flash
Running in mode: train
Using device: cuda

---> Welcome to Image-to-Image Training <---

Checking your Hardware:

-------------------------------- 
Your Hardware:

    ---> General <---
Operatingsystem: Windows
Version: 10.0.26200
Architecture: ('64bit', 'WindowsPE')
Processor: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD

    ---> GPU <---
GPU Name: NVIDIA GeForce RTX 4060
VRAM Total: 8188 MB
VRAM Used: 1039 MB
Utilization: 0.0 %
PyTorch Support: True (NVIDIA GeForce RTX 4060)
TensorFlow Support: False -> not installed

    ---> CPU <---
CPU-Name: AMD Ryzen 7 3700X 8-Core Processor
CPU Kernels: 8
Logical CPU-Kernels: 16
CPU-Frequence: 3600 MHz
CPU-Utilization: 8.6 %

    ---> RAM <---
RAM Total: 31 GB
RAM Available: 19 GB
RAM-Utilization: 38.6 %

--------------------------------
PhysGen (sound_reflection) Dataset for train got created
PhysGen (sound_reflection) Dataset for validation got created

Model Information:

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
UViT                                     [1, 1, 256, 256]          --
├─UViT: 1-1                              [1, 2, 256, 256]          197,376
│    └─PatchEmbed: 2-1                   [1, 256, 768]             --
│    │    └─Conv2d: 3-1                  [1, 768, 16, 16]          393,984
│    └─Identity: 2-2                     [1, 768]                  --
│    └─ModuleList: 2-3                   --                        --
│    │    └─Block: 3-2                   [1, 257, 768]             7,085,568
│    │    └─Block: 3-3                   [1, 257, 768]             7,085,568
│    │    └─Block: 3-4                   [1, 257, 768]             7,085,568
│    │    └─Block: 3-5                   [1, 257, 768]             7,085,568
│    │    └─Block: 3-6                   [1, 257, 768]             7,085,568
│    │    └─Block: 3-7                   [1, 257, 768]             7,085,568
│    └─Block: 2-4                        [1, 257, 768]             --
│    │    └─LayerNorm: 3-8               [1, 257, 768]             1,536
│    │    └─Attention: 3-9               [1, 257, 768]             2,360,064
│    │    └─LayerNorm: 3-10              [1, 257, 768]             1,536
│    │    └─Mlp: 3-11                    [1, 257, 768]             4,722,432
│    └─ModuleList: 2-5                   --                        --
│    │    └─Block: 3-12                  [1, 257, 768]             8,265,984
│    │    └─Block: 3-13                  [1, 257, 768]             8,265,984
│    │    └─Block: 3-14                  [1, 257, 768]             8,265,984
│    │    └─Block: 3-15                  [1, 257, 768]             8,265,984
│    │    └─Block: 3-16                  [1, 257, 768]             8,265,984
│    │    └─Block: 3-17                  [1, 257, 768]             8,265,984
│    └─LayerNorm: 2-6                    [1, 257, 768]             1,536
│    └─Linear: 2-7                       [1, 257, 512]             393,728
│    └─Conv2d: 2-8                       [1, 2, 256, 256]          38
├─Sequential: 1-2                        [1, 1, 256, 256]          --
│    └─Conv2d: 2-9                       [1, 384, 256, 256]        7,296
│    └─GELU: 2-10                        [1, 384, 256, 256]        --
│    └─Conv2d: 2-11                      [1, 384, 256, 256]        1,327,488
│    └─GELU: 2-12                        [1, 384, 256, 256]        --
│    └─Conv2d: 2-13                      [1, 1, 256, 256]          385
==========================================================================================
Total params: 101,516,711
Trainable params: 101,516,711
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 87.70
==========================================================================================
Input size (MB): 0.26
Forward/backward pass size (MB): 643.70
Params size (MB): 405.28
Estimated Total Size (MB): 1049.24
==========================================================================================


Train dataset size: 19908 | Validation dataset size: 3732

Epochs:   0%|          | 0/80 [00:00<?, ?it/s]
Epochs:   0%|          | 0/80 [06:24<?, ?it/s, time_needed=384.01s, train_loss=27.6568, val_loss=N/A]
Epochs:   1%|1         | 1/80 [06:24<8:25:40, 384.06s/it, time_needed=384.01s, train_loss=27.6568, val_loss=N/A]
Epochs:   1%|1         | 1/80 [12:43<8:25:40, 384.06s/it, time_needed=379.94s, train_loss=16.3446, val_loss=N/A]
Epochs:   2%|2         | 2/80 [12:44<8:16:08, 381.65s/it, time_needed=379.94s, train_loss=16.3446, val_loss=N/A]
Epochs:   2%|2         | 2/80 [18:53<8:16:08, 381.65s/it, time_needed=369.71s, train_loss=11.1381, val_loss=N/A]
Epochs:   4%|3         | 3/80 [18:53<8:02:48, 376.21s/it, time_needed=369.71s, train_loss=11.1381, val_loss=N/A]
Epochs:   4%|3         | 3/80 [25:02<8:02:48, 376.21s/it, time_needed=368.78s, train_loss=9.3323, val_loss=N/A] 
Epochs:   5%|5         | 4/80 [25:02<7:52:49, 373.28s/it, time_needed=368.78s, train_loss=9.3323, val_loss=N/A]
Epochs:   5%|5         | 4/80 [31:53<7:52:49, 373.28s/it, time_needed=366.22s, train_loss=9.1186, val_loss=8.3822]
Epochs:   6%|6         | 5/80 [32:07<8:09:48, 391.85s/it, time_needed=366.22s, train_loss=9.1186, val_loss=8.3822]
```



<br>
<br>

---
### Process

1. Clone/download this repository
2. Run [the installation](#installation)
3. Open for example anaconda prompt / bash and navigate to your project folder
    ```bash
    cd "D:\Studium\Master\Repos\Image-to-Image"
    D:
    ```
4. Activate the conda env:
    ```bash
    conda activate img-to-img
    ```
5. Maybe check your hardware with:
    ```bash
    nvidia-smi
    ```
6. Preparation -> Folder creation
    ```bash
    mkdir logs
    ```
7. Run Train script with logging:
    ```bash
    nohup python ./image-to-image/main.py \
    --mode train \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --loss l1 \
    --optimizer adam \
    --weight_decay \
    --weight_decay_rate 0.0004 \
    --gradient_clipping \
    --gradient_clipping_threshold 0.5 \
    --scheduler step \
    --activate_amp \
    --amp_scaler grad \
    --checkpoint_save_dir ./checkpoints \
    --save_only_best_model \
    --validation_interval 5 \
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
    --cmap gray \
    > ./logs/resfcn_test.log 2>&1 &
    ```
    or at windows:
    ```bash
    start /B python ./image-to-image/main.py `
    --mode train `
    --epochs 100 `
    --batch_size 16 `
    --lr 0.0001 `
    --loss l1 `
    --optimizer adam `
    --weight_decay `
    --weight_decay_rate 0.0004 `
    --gradient_clipping `
    --gradient_clipping_threshold 0.5 `
    --scheduler step `
    --activate_amp `
    --amp_scaler grad `
    --checkpoint_save_dir ./checkpoints `
    --save_only_best_model `
    --validation_interval 5 ^
    --model resfcn `
    --resfcn_in_channels 1 `
    --resfcn_hidden_channels 64 `
    --resfcn_out_channels 1 `
    --resfcn_num_blocks 16 `
    --data_variation sound_reflection `
    --input_type osm `
    --output_type standard `
    --device cuda `
    --experiment_name image-to-image `
    --run_name resfcn_test `
    --tensorboard_path ./tensorboard `
    --save_path ./mlflow_images `
    --cmap gray `
    > ./logs/resfcn_test.log 2>&1
    ```
8. Now check you log-file and in worse case stop the run:<br>
    Linux:
    ```bash
    ps aux | grep '[p]ython'
    kill -9 -f python
    ```
    Windows CMD:
    ```bash
    tasklist /FI "IMAGENAME eq python.exe"
    taskkill /IM python.exe /F
    ```
    Windows PowerShell:
    ```bash
    Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*python*" } | Select-Object ProcessId, CommandLine
    Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*python*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    ```
9. If training is finish, you can open the `. > image_to_image > model_interactions > eval_physgen_benchmark.ipynb` ([or click me](./image_to_image/model_interactions/eval_physgen_benchmark.ipynb)), change the variables and run the code blocks to evaluate your model.
10. You can also check your mlrun/tensorboard metrics, to find out how the training was -> [see here](#access-experiment-tracking)


> When pasting multiline commands on windows, use \` for splitting commands when using Windows PowerShell, for CMD or CMD-kinds use `^` and for bash/linux like command terminales you might want to try `\`.

> See also the [Google Coolab notebook](./google_coolab.ipynb).

<br>
<br>

---
### Example Commands

**Training run**
```bash
python ./main.py \
  --mode train \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --loss l1 \
  --optimizer adam \
  --weight_decay \
  --weight_decay_rate 0.0004 \
  --gradient_clipping \
  --gradient_clipping_threshold 0.5 \
  --scheduler step \
  --activate_amp \
  --amp_scaler grad \
  --checkpoint_save_dir ./checkpoints \
  --save_only_best_model \
  --validation_interval 5 \
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
python ./main.py \
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
python ./main.py \
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
python ./main.py \
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


> In windows you have to use ``` or `^` for the line breaks or remove the line breaks completly. Or you use Linux via Docker.

<br>
<br>

---
### Experiment Commands

**Pix2Pix Rebuild**

Linux:
```bash
nohup python ./main.py \
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
  --optimizer_2 adam \
  --weight_decay \
  --weight_decay_rate 0.0004 \
  --gradient_clipping \
  --gradient_clipping_threshold 0.5 \
  --scheduler step \
  --scheduler_2 step \
  --activate_amp \
  --amp_scaler grad \
  --checkpoint_save_dir ./checkpoints \
  --save_only_best_model \
  --validation_interval 5 \
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
  --cmap gray \
  > ./logs/pix2pix_rebuild_test.log 2>&1 &
```

Windows:
```bash
start /B python ./main.py ^
  --mode train ^
  --epochs 100 ^
  --batch_size 16 ^
  --lr 0.0001 ^
  --loss weighted_combined ^
  --wc_loss_silog_lambda 0.5 ^
  --wc_loss_weight_silog 1.0 ^
  --wc_loss_weight_grad 50.0 ^
  --wc_loss_weight_ssim 100.0 ^
  --wc_loss_weight_edge_aware 50.0 ^
  --wc_loss_weight_l1 10.0 ^
  --wc_loss_weight_var 0.0 ^
  --wc_loss_weight_range 0.0 ^
  --wc_loss_weight_blur 0.0 ^
  --optimizer adam ^
  --optimizer_2 adam ^
  --weight_decay ^
  --weight_decay_rate 0.0004 ^
  --gradient_clipping ^
  --gradient_clipping_threshold 0.5 ^
  --scheduler step ^
  --scheduler_2 step ^
  --activate_amp ^
  --amp_scaler grad ^
  --checkpoint_save_dir ./checkpoints ^
  --save_only_best_model ^
  --validation_interval 5 ^
  --model pix2pix ^
  --pix2pix_in_channels 1 ^
  --pix2pix_hidden_channels 64 ^
  --pix2pix_out_channels 1 ^
  --pix2pix_second_loss_lambda 100 ^
  --data_variation sound_reflection ^
  --input_type osm ^
  --output_type standard ^
  --device cuda ^
  --experiment_name image-to-image ^
  --run_name pix2pix_rebuild_test ^
  --tensorboard_path ./tensorboard ^
  --save_path ./mlflow_images ^
  --cmap gray ^
  > ./logs/pix2pix_rebuild_test.log 2>&1
```

```bash
start /B python ./main.py ^
  --mode train ^
  --epochs 50 ^
  --batch_size 16 ^
  --lr 0.0005 ^
  --loss weighted_combined ^
  --wc_loss_silog_lambda 0.5 ^
  --wc_loss_weight_silog 0.1 ^
  --wc_loss_weight_grad 5.0 ^
  --wc_loss_weight_ssim 1.0 ^
  --wc_loss_weight_edge_aware 5.0 ^
  --wc_loss_weight_l1 1.0 ^
  --wc_loss_weight_var 0.0 ^
  --wc_loss_weight_range 0.0 ^
  --wc_loss_weight_blur 0.0 ^
  --optimizer adamw ^
  --optimizer_2 adam ^
  --weight_decay ^
  --weight_decay_rate 0.01 ^
  --gradient_clipping ^
  --gradient_clipping_threshold 2.0 ^
  --scheduler cosine ^
  --scheduler_2 step ^
  --activate_amp ^
  --amp_scaler grad ^
  --checkpoint_save_dir ./checkpoints ^
  --save_only_best_model ^
  --validation_interval 1 ^
  --model pix2pix ^
  --pix2pix_in_channels 1 ^
  --pix2pix_hidden_channels 64 ^
  --pix2pix_out_channels 1 ^
  --pix2pix_second_loss_lambda 500.0 ^
  --data_variation sound_reflection ^
  --input_type osm ^
  --output_type standard ^
  --device cuda ^
  --experiment_name image-to-image ^
  --run_name pix2pix_rebuild_test ^
  --tensorboard_path ./tensorboard ^
  --save_path ./mlflow_images ^
  --cmap gray ^
  > ./logs/pix2pix_rebuild_test.log 2>&1
```



<br><br>

**New Simple Model try out**

Linux:
```bash
nohub python ./main.py \
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
  --weight_decay \
  --weight_decay_rate 0.0004 \
  --gradient_clipping \
  --gradient_clipping_threshold 0.5 \
  --scheduler step \
  --activate_amp \
  --amp_scaler grad \
  --checkpoint_save_dir ./checkpoints \
  --save_only_best_model \
  --validation_interval 5 \
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
  --cmap gray \
  > ./logs/resfcn_weighted_combined_test.log 2>&1 &
```

Windows:
```bash
start /B python ./main.py ^
  --mode train ^
  --epochs 50 ^
  --batch_size 16 ^
  --lr 0.001 ^
  --loss weighted_combined ^
  --wc_loss_silog_lambda 0.5 ^
  --wc_loss_weight_silog 0.1 ^
  --wc_loss_weight_grad 5.0 ^
  --wc_loss_weight_ssim 10.0 ^
  --wc_loss_weight_edge_aware 5.0 ^
  --wc_loss_weight_l1 1.0 ^
  --wc_loss_weight_var 0.0 ^
  --wc_loss_weight_range 0.0 ^
  --wc_loss_weight_blur 0.0 ^
  --optimizer adamw ^
  --weight_decay ^
  --weight_decay_rate 0.05 ^
  --gradient_clipping ^
  --gradient_clipping_threshold 2.0 ^
  --scheduler step ^
  --activate_amp ^
  --amp_scaler grad ^
  --checkpoint_save_dir ./checkpoints ^
  --save_only_best_model ^
  --validation_interval 2 ^
  --model resfcn ^
  --resfcn_in_channels 1 ^
  --resfcn_hidden_channels 64 ^
  --resfcn_out_channels 1 ^
  --resfcn_num_blocks 16 ^
  --data_variation sound_reflection ^
  --input_type osm ^
  --output_type standard ^
  --device cuda ^
  --experiment_name image-to-image ^
  --run_name resfcn_weighted_combined_test ^
  --tensorboard_path ./tensorboard ^
  --save_path ./mlflow_images ^
  --cmap gray ^
  > ./logs/resfcn_weighted_combined_test.log 2>&1
```

<br><br>

**Residual Design Model Test**

Linux:
```bash
nohup python ./main.py \
  --mode train \
  --epochs 120 \
  --batch_size 12 \
  --lr 0.0001 \
  --loss weighted_combined \
  --wc_loss_silog_lambda 0.5 \
  --wc_loss_weight_silog 1.0 \
  --wc_loss_weight_grad 40.0 \
  --wc_loss_weight_ssim 80.0 \
  --wc_loss_weight_edge_aware 40.0 \
  --wc_loss_weight_l1 8.0 \
  --wc_loss_weight_var 0.0 \
  --wc_loss_weight_range 0.0 \
  --wc_loss_weight_blur 0.0 \
  --optimizer adam \
  --optimizer_2 adamw \
  --weight_decay \
  --weight_decay_rate 0.0004 \
  --gradient_clipping \
  --gradient_clipping_threshold 0.5 \
  --scheduler step \
  --scheduler_2 step \
  --activate_amp \
  --amp_scaler grad \
  --checkpoint_save_dir ./checkpoints \
  --save_only_best_model \
  --validation_interval 5 \
  --model residual_design_model \
  --base_model pix2pix \
  --complex_model resfcn \
  --combine_mode nn \
  --loss_2 weighted_combined \
  --wc_loss_silog_lambda_2 0.5 \
  --wc_loss_weight_silog_2 1.0 \
  --wc_loss_weight_grad_2 60.0 \
  --wc_loss_weight_ssim_2 30.0 \
  --wc_loss_weight_edge_aware_2 60.0 \
  --wc_loss_weight_l1_2 6.0 \
  --resfcn_2_in_channels 1 \
  --resfcn_2_hidden_channels 64 \
  --resfcn_2_out_channels 1 \
  --resfcn_2_num_blocks 12 \
  --pix2pix_in_channels 1 \
  --pix2pix_hidden_channels 64 \
  --pix2pix_out_channels 1 \
  --pix2pix_second_loss_lambda 100 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --device cuda \
  --experiment_name image-to-image \
  --run_name residual_design_nn_test \
  --tensorboard_path ./tensorboard \
  --save_path ./mlflow_images \
  --cmap gray \
  > ./logs/residual_design_nn_test.log 2>&1 &
```

Windows:
```bash
start /B python ./main.py ^
  --mode train ^
  --epochs 120 ^
  --batch_size 12 ^
  --lr 0.0001 ^
  --loss weighted_combined ^
  --wc_loss_silog_lambda 0.5 ^
  --wc_loss_weight_silog 1.0 ^
  --wc_loss_weight_grad 40.0 ^
  --wc_loss_weight_ssim 80.0 ^
  --wc_loss_weight_edge_aware 40.0 ^
  --wc_loss_weight_l1 8.0 ^
  --wc_loss_weight_var 0.0 ^
  --wc_loss_weight_range 0.0 ^
  --wc_loss_weight_blur 0.0 ^
  --optimizer adam ^
  --optimizer_2 adamw ^
  --weight_decay ^
  --weight_decay_rate 0.001 ^
  --gradient_clipping ^
  --gradient_clipping_threshold 0.5 ^
  --scheduler step ^
  --scheduler_2 step ^
  --activate_amp ^
  --amp_scaler grad ^
  --checkpoint_save_dir ./checkpoints ^
  --save_only_best_model ^
  --validation_interval 5 ^
  --model residual_design_model ^
  --base_model pix2pix ^
  --complex_model resfcn ^
  --combine_mode nn ^
  --loss_2 weighted_combined ^
  --wc_loss_silog_lambda_2 0.5 ^
  --wc_loss_weight_silog_2 1.0 ^
  --wc_loss_weight_grad_2 60.0 ^
  --wc_loss_weight_ssim_2 30.0 ^
  --wc_loss_weight_edge_aware_2 60.0 ^
  --wc_loss_weight_l1_2 6.0 ^
  --resfcn_2_in_channels 1 ^
  --resfcn_2_hidden_channels 64 ^
  --resfcn_2_out_channels 1 ^
  --resfcn_2_num_blocks 12 ^
  --pix2pix_in_channels 1 ^
  --pix2pix_hidden_channels 64 ^
  --pix2pix_out_channels 1 ^
  --pix2pix_second_loss_lambda 100 ^
  --data_variation sound_reflection ^
  --input_type osm ^
  --output_type standard ^
  --device cuda ^
  --experiment_name image-to-image ^
  --run_name residual_design_nn_test ^
  --tensorboard_path ./tensorboard ^
  --save_path ./mlflow_images ^
  --cmap gray ^
  > ./logs/residual_design_nn_test.log 2>&1
```

<br><br>

**PhysFormer tryout**

Linux:
```bash
nohub python ./main.py \
  --mode train \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.001 \
  --loss weighted_combined \
  --wc_loss_silog_lambda 0.5 \
  --wc_loss_weight_silog 0.1 \
  --wc_loss_weight_grad 5.0 \
  --wc_loss_weight_ssim 10.0 \
  --wc_loss_weight_edge_aware 5.0 \
  --wc_loss_weight_l1 1.0 \
  --wc_loss_weight_var 0.0 \
  --wc_loss_weight_range 0.0 \
  --wc_loss_weight_blur 0.0 \
  --optimizer adamw \
  --weight_decay \
  --weight_decay_rate 0.05 \
  --gradient_clipping \
  --gradient_clipping_threshold 2.0 \
  --scheduler cosine \
  --use_warm_up \
  --warm_up_start_lr 0.000005 \
  --warm_up_step_duration 2000 \
  --activate_amp \
  --amp_scaler grad \
  --checkpoint_save_dir ./checkpoints \
  --save_only_best_model \
  --validation_interval 2 \
  --model physicsformer \
  --physicsformer_in_channels 1 \
  --physicsformer_out_channels 1 \
  --physicsformer_img_size 256 \
  --physicsformer_patch_size 4 \
  --physicsformer_embedded_dim 1024 \
  --physicsformer_num_blocks 8 \
  --physicsformer_heads 16 \
  --physicsformer_mlp_dim 2048 \
  --physicsformer_dropout 0.1 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --device cuda \
  --experiment_name image-to-image \
  --run_name physformer_test \
  --tensorboard_path ./tensorboard \
  --save_path ./mlflow_images \
  --cmap gray \
  > ./logs/physformer_test.log 2>&1
```

Windows:
```bash
start /B python ./main.py ^
  --mode train ^
  --epochs 50 ^
  --batch_size 16 ^
  --lr 0.001 ^
  --loss weighted_combined ^
  --wc_loss_silog_lambda 0.5 ^
  --wc_loss_weight_silog 0.1 ^
  --wc_loss_weight_grad 5.0 ^
  --wc_loss_weight_ssim 10.0 ^
  --wc_loss_weight_edge_aware 5.0 ^
  --wc_loss_weight_l1 1.0 ^
  --wc_loss_weight_var 0.0 ^
  --wc_loss_weight_range 0.0 ^
  --wc_loss_weight_blur 0.0 ^
  --optimizer adamw ^
  --weight_decay ^
  --weight_decay_rate 0.05 ^
  --gradient_clipping ^
  --gradient_clipping_threshold 2.0 ^
  --scheduler cosine ^
  --use_warm_up ^
  --warm_up_start_lr 0.000005 ^
  --warm_up_step_duration 2000 ^
  --activate_amp ^
  --amp_scaler grad ^
  --checkpoint_save_dir ./checkpoints ^
  --save_only_best_model ^
  --validation_interval 2 ^
  --model physicsformer ^
  --physicsformer_in_channels 1 ^
  --physicsformer_out_channels 1 ^
  --physicsformer_img_size 256 ^
  --physicsformer_patch_size 8 ^
  --physicsformer_embedded_dim 1024 ^
  --physicsformer_num_blocks 8 ^
  --physicsformer_heads 16 ^
  --physicsformer_mlp_dim 2048 ^
  --physicsformer_dropout 0.1 ^
  --data_variation sound_reflection ^
  --input_type osm ^
  --output_type standard ^
  --device cuda ^
  --experiment_name image-to-image ^
  --run_name physicsformer_test ^
  --tensorboard_path ./tensorboard ^
  --save_path ./mlflow_images ^
  --cmap gray ^
  > ./logs/physicsformer_test.log 2>&1
```

<br><br>

**Ray-Tracing Test**

Linux:
```bash
nohub python ./main.py \
  --mode train \
  --epochs 120 \
  --batch_size 12 \
  --lr 0.0001 \
  --loss weighted_combined \
  --wc_loss_silog_lambda 0.5 \
  --wc_loss_weight_silog 1.0 \
  --wc_loss_weight_grad 40.0 \
  --wc_loss_weight_ssim 80.0 \
  --wc_loss_weight_edge_aware 40.0 \
  --wc_loss_weight_l1 8.0 \
  --wc_loss_weight_var 0.0 \
  --wc_loss_weight_range 0.0 \
  --wc_loss_weight_blur 0.0 \
  --optimizer adam \
  --optimizer_2 adamw \
  --weight_decay \
  --weight_decay_rate 0.0004 \
  --gradient_clipping \
  --gradient_clipping_threshold 0.5 \
  --scheduler step \
  --scheduler_2 step \
  --activate_amp \
  --amp_scaler grad \
  --checkpoint_save_dir ./checkpoints \
  --save_only_best_model \
  --validation_interval 5 \
  --model residual_design_model \
  --base_model pix2pix \
  --complex_model pix2pix \
  --combine_mode nn \
  --loss_2 weighted_combined \
  --wc_loss_silog_lambda_2 0.5 \
  --wc_loss_weight_silog_2 1.0 \
  --wc_loss_weight_grad_2 60.0 \
  --wc_loss_weight_ssim_2 30.0 \
  --wc_loss_weight_edge_aware_2 60.0 \
  --wc_loss_weight_l1_2 6.0 \
  --pix2pix_in_channels 1 \
  --pix2pix_hidden_channels 64 \
  --pix2pix_out_channels 1 \
  --pix2pix_second_loss_lambda 100 \
  --pix2pix_2_in_channels 2 \
  --pix2pix_2_hidden_channels 64 \
  --pix2pix_2_out_channels 1 \
  --pix2pix_2_second_loss_lambda 100 \
  --data_variation sound_reflection \
  --input_type osm \
  --output_type standard \
  --reflexion_channels \
  --reflexion_steps 36 \
  --device cuda \
  --experiment_name image-to-image \
  --run_name raytracing_test \
  --tensorboard_path ./tensorboard \
  --save_path ./mlflow_images \
  --cmap gray \
  > ./logs/raytracing_test.log 2>&1 &
```

Windows:
```bash
start /B python ./main.py ^
  --mode train ^
  --epochs 120 ^
  --batch_size 12 ^
  --lr 0.0001 ^
  --loss weighted_combined ^
  --wc_loss_silog_lambda 0.5 ^
  --wc_loss_weight_silog 1.0 ^
  --wc_loss_weight_grad 40.0 ^
  --wc_loss_weight_ssim 80.0 ^
  --wc_loss_weight_edge_aware 40.0 ^
  --wc_loss_weight_l1 8.0 ^
  --wc_loss_weight_var 0.0 ^
  --wc_loss_weight_range 0.0 ^
  --wc_loss_weight_blur 0.0 ^
  --optimizer adam ^
  --optimizer_2 adamw ^
  --weight_decay ^
  --weight_decay_rate 0.0004 ^
  --gradient_clipping ^
  --gradient_clipping_threshold 0.5 ^
  --scheduler step ^
  --scheduler_2 step ^
  --activate_amp ^
  --amp_scaler grad ^
  --checkpoint_save_dir ./checkpoints ^
  --save_only_best_model ^
  --validation_interval 5 ^
  --model residual_design_model ^
  --base_model pix2pix ^
  --complex_model pix2pix ^
  --combine_mode nn ^
  --loss_2 weighted_combined ^
  --wc_loss_silog_lambda_2 0.5 ^
  --wc_loss_weight_silog_2 1.0 ^
  --wc_loss_weight_grad_2 60.0 ^
  --wc_loss_weight_ssim_2 30.0 ^
  --wc_loss_weight_edge_aware_2 60.0 ^
  --wc_loss_weight_l1_2 6.0 ^
  --pix2pix_in_channels 1 ^
  --pix2pix_hidden_channels 64 ^
  --pix2pix_out_channels 1 ^
  --pix2pix_second_loss_lambda 100 ^
  --pix2pix_2_in_channels 2 ^
  --pix2pix_2_hidden_channels 64 ^
  --pix2pix_2_out_channels 1 ^
  --pix2pix_2_second_loss_lambda 100 ^
  --data_variation sound_reflection ^
  --input_type osm ^
  --output_type standard ^
  --reflexion_channels ^
  --reflexion_steps 36 ^
  --device cuda ^
  --experiment_name image-to-image ^
  --run_name raytracing_test ^
  --tensorboard_path ./tensorboard ^
  --save_path ./mlflow_images ^
  --cmap gray ^
  > ./logs/raytracing_test.log 2>&1 &
```

<br><br>

**U-ViT Test**


Windows:
```bash
start /B python ./main.py ^
  --mode train ^
  --epochs 80 ^
  --batch_size 32 ^
  --lr 0.0001 ^
  --loss weighted_combined ^
  --wc_loss_silog_lambda 0.5 ^
  --wc_loss_weight_silog 0.0 ^
  --wc_loss_weight_grad 0.0 ^
  --wc_loss_weight_ssim 10.0 ^
  --wc_loss_weight_edge_aware 0.0 ^
  --wc_loss_weight_l1 100.0 ^
  --wc_loss_weight_var 0.0 ^
  --wc_loss_weight_range 0.0 ^
  --wc_loss_weight_blur 10.0 ^
  --optimizer adam ^
  --weight_decay ^
  --weight_decay_rate 0.000004 ^
  --gradient_clipping ^
  --gradient_clipping_threshold 0.5 ^
  --scheduler cosine ^
  --use_warm_up ^
  --warm_up_start_lr 0.000005 ^
  --warm_up_step_duration 2000 ^
  --activate_amp ^
  --amp_scaler grad ^
  --checkpoint_save_dir ./checkpoints ^
  --save_only_best_model ^
  --validation_interval 3 ^
  --model uvit ^
  --uvit_in_channels 1 ^
  --uvit_hidden_channels 768 ^
  --uvit_out_channels 1 ^
  --uvit_image_size 256 ^
  --uvit_timesteps 500 ^
  --data_variation sound_reflection ^
  --input_type osm ^
  --output_type standard ^
  --device cuda ^
  --experiment_name image-to-image ^
  --run_name uvit_test ^
  --tensorboard_path ./tensorboard ^
  --save_path ./mlflow_images ^
  --cmap gray ^
  > ./logs/uvit_test.log 2>&1 &
```

taskkill /IM python.exe /F


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



