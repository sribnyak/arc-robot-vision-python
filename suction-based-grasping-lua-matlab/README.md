# Robotic Pick-and-Place of Novel Objects in Clutter: Suction-Based Grasping

A Torch implementation of fully convolutional neural networks for predicting pixel-level affordances (here higher values indicate better surface locations for grasping with suction) given an RGB-D image as input.

## Requirements and Dependencies

* NVIDIA GPU with compute capability 3.5+
* [Torch](http://torch.ch/) with packages: [image](https://github.com/torch/image), [optim](https://github.com/torch/optim), [inn](https://github.com/szagoruyko/imagine-nn), [cutorch](https://github.com/torch/cutorch), [cunn](https://github.com/torch/cunn), [cudnn](https://github.com/soumith/cudnn.torch), [hdf5](https://github.com/deepmind/torch-hdf5)
* [Matlab](https://www.mathworks.com/products/matlab.html) 2015b or later

Our implementations have been tested on Ubuntu 16.04 with an NVIDIA Titan X. Our full pick-and-place system implementation (outside the scope of this repository) uses a lightweight C++ ROS service as a wrapper to control Torch/Lua and Matlab processes via TCP sockets. Data is shared between the processes by reading and writing from RAMDisk.

## Quick Start

To run our pre-trained model to get pixel-level affordances for grasping with suction:

1. Clone this repository and navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    git clone https://github.com/andyzeng/arc-robot-vision.git
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. Download our pre-trained model for suction-based grasping:

    ```bash
    wget http://3dvision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7
    ```

    Direct download link: [suction-based-grasping-snapshot-10001.t7 (450.1 MB)](http://3dvision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7)

3. Run our model on an optional target RGB-D image. Input color images should be 24-bit RGB PNG, while depth images should be 16-bit PNG, where depth values are saved in deci-millimeters (10<sup>-4</sup>m).

    ```bash
    th infer.lua # creates results.h5
    ```

    or

    ```bash
    imgColorPath=<image.png> imgDepthPath=<image.png> modelPath=<model.t7> th infer.lua # creates results.h5
    ```

4. Visualize the predictions in Matlab. Shows a heat map of confidence values where hotter regions indicate better locations for grasping with suction. Also displays computed surface normals, which can be used to decide between robot motion primitives suction-down or suction-side. Run the following in Matlab:

    ```matlab
    visualize; % creates results.png and normals.png
    ```

## Training

To train your own model:

1. Navigate to `arc-robot-vision/suction-based-grasping`

    ```bash
    cd arc-robot-vision/suction-based-grasping
    ```

2. Download our suction-based grasping dataset and save the files into `arc-robot-vision/suction-based-grasping/data`. More information about the dataset can be found [here](http://3dvision.princeton.edu/projects/2017/arc/#datasets).

    ```bash
    wget http://3dvision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip
    unzip suction-based-grasping-dataset.zip # unzip dataset
    ```

    Direct download link: [suction-based-grasping-dataset.zip (1.6 GB)](http://3dvision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip)

3. Download the Torch ResNet-101 model pre-trained on ImageNet:

    ```bash
    cd convnet
    wget http://3dvision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7
    ```

    Direct download link: [resnet-101.t7 (409.4 MB)](http://3dvision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7)

4. Run training (set optional parameters through command line arguments):

    ```bash
    th train.lua
     ```

    Tip: if you run out of GPU memory (CUDA error=2), reduce batch size or modify the network architecture in `model.lua` to use the smaller [ResNet-50 (256.7 MB)](http://3dvision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7) model pre-trained on ImageNet.

## Evaluation

To evaluate a trained model:

1. Navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. Run our pre-trained model to get affordance predictions for the testing split of our grasping dataset:

    ```bash
    th test.lua # creates evaluation-results.h5
    ```

    or run your own model:

    ```bash
    modelPath=<model.t7> th test.lua # creates evaluation-results.h5
    ```

3. Run the evaluation script in Matlab to compute pixel-level precision against manual annotations from the grasping dataset, as reported in our [paper](https://arxiv.org/pdf/1710.01330.pdf):

    ```matlab
    evaluate;
    ```

## Baseline Algorithm

Our baseline algorithm predicts affordances for suction-based grasping by first computing 3D surface normals of the point cloud (projected from the RGB-D image), then measuring the variance of the surface normals (higher variance = lower affordance). To run our baseline algorithm over the testing split of our grasping dataset:

1. Navigate to `arc-robot-vision/suction-based-grasping/baseline`

    ```bash
    cd arc-robot-vision/suction-based-grasping/baseline
    ```

2. Run the following in Matlab:

    ```matlab
    test; % creates results.mat
    evaluate;
    ```
