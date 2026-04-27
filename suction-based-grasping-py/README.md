## Code structure

`convnet/` - MIT-Princeton solution for suction-based grasping using a CNN:
- `demo/` - test images and camera intrinsics
- `experiments.ipynb` - a notebook for interactive code testing and playing
- `dataset.py` - the dataset class, corresponds to original `DataLoader.lua`
- `model.py` - model architecture
- `train.py` - model training 
- `test.py` - model testing
- `infer.py` - model inference
- `model_utils.py` - utility functions for models
- `spatial_symmetric_padding.py` - SpatialSymmetricPadding module
- `util.py` - utility functions
- `postprocess.py` - post-process affordance maps with background subtraction and removing regions with high variance in 3D surface normals
- `evaluate.py` - evaluating suction-based grasping affordance predictions
- `visualize.py` - post-processing and visualizing suction-based grasping affordance predictions

## Run

1. Prepare a python environment, install requirements from requirements.txt
2. Download [the dataset for training](http://3dvision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip) and extract it into `suction-based-grasping-py/`
