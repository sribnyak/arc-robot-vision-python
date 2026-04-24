## Code structure

`convnet/` - MIT-Princeton solution for suction-based grasping using a CNN:
- `demo/` - test images and camera intrinsics
- `dataloader.py` - defines the dataloader
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
