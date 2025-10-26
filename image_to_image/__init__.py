"""
The Image-to-Image package consists of 
PyTorch models, run scripts (training and inference),
Physgen Dataloader, Testscript and evaluation notebook.
It also adds special losses, scheduler and more.

Implemented models:
- Pix2Pix (UNet Generator with adversarial loss)
- PhysFormer (Transformer)
- ResFCN (Simple fully convolutional network - no shrinking)
- ResidualDesignModel (Model which consist of 2 of the other models)

Use this model by running:
```python
import sys
sys.path += ["."]

from image_to_image import main
main()
```

Now set all your arguments/settings through the python argument. Example:
```
start /B python ./your_py_file_name.py ^
  --mode train ^
  --epochs 50 ^
  --batch_size 1 ^
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
  --physicsformer_patch_size 4 ^
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

Open the folder with the evaluation notebook:
```python
from image_to_image import open_notebook_folder
open_notebook_folder()
```
"""
from . import model_interactions
from . import utils
from .main import main
from .utils.package import open_notebook_folder
