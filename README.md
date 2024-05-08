# Video Frame Interpolation via Adaptive Sepearable Convolution

This is our implementation of the work of Niklaus et al.'s Adaptive Separable Convolution method for video frame interpolation \[[1](#references)\]. we explore the efficacy of this technique by applying the same network structure to a smaller dataset and experimenting with various loss functions.

## Installation

Our project requires Python 3.11 and pytorch 2.2.2 (with cuda). Although all requirements are mentioned in the `requirements.txt`, you might need to install torch and torchvision for cuda separately.

To install all dependencies, run the command:
>`pip install -r requirements.txt`

## Running the project

Our training code is present in the `train.py` file. Our testing code is present in the `test.py` file. The `enhance.py` contains the code for increasing the framerate of any input video.

To run any of these files, run the command:
>`python <filename>.py`

## Acknowledgement

We have modified the cuda kernel code already present in [HyeongminLEE's implementation](https://github.com/HyeongminLEE/pytorch-sepconv) to be compatible with the latest version of cupy.

## References

<!-- <div id="references"></div> -->
\[1\] Video Frame Interpolation via Adaptive Separable Convolution, Simon Niklaus, Long Mai, Feng Liu (2017) [arXiv:1708.01692](https://arxiv.org/abs/1708.01692)