## Code structure

Implemented:
`convnet/` - MIT-Princeton solution for suction-based grasping using a CNN:
- `demo/` - test images and camera intrinsics
- `experiments.ipynb` - a notebook for interactive code testing and playing
- `dataset.py` - the dataset class, corresponds to original `DataLoader.lua`
- `model.py` - model architecture
- `metrics.py` - metrics and losses
- `infer.py` - model inference
- `train.py` - model training

To be implemented:
- `postprocess.py` - post-process affordance maps with background subtraction and removing regions with high variance in 3D surface normals
- `evaluate.py` - evaluating suction-based grasping affordance predictions
- `visualize.py` - post-processing and visualizing suction-based grasping affordance predictions

## Setup

1. Activate a python environment, python version from 3.10 to 3.13
2. Install PyTorch + CUDA compatible with your GPU:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
3. Install remaining dependencies:
```bash
pip3 install -r suction-based-grasping-py/requirements.txt
```

## Inference

```bash
cd suction-based-grasping-py/convnet/
python3 infer.py [arguments]
```

## Training

1. Download [the dataset for training](http://3dvision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip) and extract it into `suction-based-grasping-py`
2. Run training (arguments override config values, e.g. `device=cpu`)
```bash
cd suction-based-grasping-py/convnet/
python3 train.py [arguments]
```
Use tmux to prevent training from sudden interruptions:
```bash
tmux new -s train           # start a session
python3 train.py [arguments]
# detach: Ctrl-B D
tmux ls                     # list sessions
tmux attach -t train        # reattach
tmux kill-session -t train  # stop/kill session
```
3. To see training progress, open TensorBoard (in another terminal):
```bash
cd suction-based-grasping-py/convnet/
tensorboard --logdir logs
```
