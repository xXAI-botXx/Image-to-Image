# Image-to-Image
An Image-to-Image PyTorch repo.

### Installation

```bash
conda create -n img-to-img python=3.13 pip -y
conda activate img-to-img
pip install -r requirements.txt

torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Use your CUDA needed. You may want to check with `nvidia-smi`.

### Example Commands

**Training run**
```bash
FIXME
python -m main.py \
  --mode train \
  --variation sound_baseline \
  --data_dir ./datasets/trainset \
  --save_dir ./checkpoints \
  --epochs 100 --batch_size 16 --lr 0.0001 \
```


**Testing**
```bash
FIXME

python -m main.py \
  --mode test \
  --checkpoint ./checkpoints/epoch_50.pth
```


**Inference**
```bash
FIXME
python -m main.py \
  --mode inference \
  --input_dir ./samples \
  --output_dir ./results \
  --checkpoint ./checkpoints/best_model.pth
```



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



