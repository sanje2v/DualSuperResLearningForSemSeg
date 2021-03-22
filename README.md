# Implementation of 'Dual Super Resolution Learning For Semantic Segmentation', CVPR 2020 paper
[This paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.html) combines Super-Resolution and Feature Affinity learning to improve traditional semantic segmentation model.


![picture](demo/stage1_output.png)

# Results
On *256x512* input with *512x1024* output segmentation map, trained with *250* epochs:

| Type        | Mean Accuracy % | Mean IoU %            | Cross Entropy Error | Best Epoch |
| ------------|:---------------:|:---------------------:|:-------------------:|-----------:|
|SSSR         |93.28            |57.83 (51.78)          |0.228                |250         |
|SSSR+SISR    |**93.48**        |60.59 (53.21)          |**0.224**            |248         |
|SSSR+SISR+FA |93.34            |**60.96 (53.33)**      |0.227                |**234**     |
*NOTE: The reported mean IoU is computed using sum of intersections divided by sum of intersections (popular usage) while the ones in bracket is the mean of individually computed intersections over unions (precise definition of Mean IoU).*

# Things to know
* '.weights' file is to be used for inference and ONLY contains weights for Stage 1 layers except for 'final.weights' produced at the end of training a stage. This file contains weights for all the network layers introduced in that stage and below it.
* '.checkpoint' file contains, on top of Stage 1 weights, other information such as optimizer state, epochs etc that can be used to resume training. 'Inspect checkpoint' discussed below shows how to print these values.
* '.cpkt' file is weights just like '.weights' file downloaded from model zoo but doesn't contain any dictionary.

# Requirements
The following software versions were used for testing the code in this repo. Other version combination of software might also work but have not been tested on.
* Python 3.7
* PyTorch 1.7
* TorchVision 0.8.1
* CUDA 11.1
* Conda 4.9.2
* Microsoft Visual Studio 2019 (if using .sln file)
* Required python libraries are in 'requirements.txt'


# Getting started
Before training or benchmarking, please download 'gtFine.zip' and 'leftImg8bit.zip' for [**Cityscapes dataset**](https://www.cityscapes-dataset.com/) and unzip them under './datasets/Cityscapes/data'.

All commands are invoked via `main.py` script. For instance to train, you would use something like `python main.py --train [...]`. Use `python main.py --help` to view all supported actions.


# Commands
## Training
Perform training on the '*train*' split of **Cityscapes dataset**.

*Example usage:*
`python main.py train --stage 1 --description "Stage 1 training" --epochs 200 --batch-size 6 --device gpu --checkpoint-history 40 --num-workers 4 --val-interval 5` 


## Resume training
Resume training from an interrupted session using checkpoint file (that have '.checkpoint' extension and are autosaved under './weights/checkpoints').

*Example usage:*
`python main.py resume-train --checkpoint ./weights/checkpoints/epoch50.checkpoint`


## Testing
Perform inference on a specified image file with specified weights and shows result as well as saves it to a file.

*Example usage:*
`python main.py test --image-file ~/Pictures/input.jpg --weights ./weights/stage3/final.weights --device gpu`

The result of the inference is saved in `./outputs/<image_filename>.png`.


## Weights pruning
Weights trained in stage 2 and 3 will have weights for parts of model not needed for inference. So, weights pruning will remove all these weights.

*Example usage:*
`python main.py prune-weights --src-weights ./weights/stage3/checkpoints/epoch50.checkpoint --dest-weights ./output/inference.weights`


## Inspect checkpoint
View the keys and values dictionary pairs in the specified checkpoint file.

*Example usage:*
`python main.py inspect-checkpoint --checkpoint ./weights/stage2/checkpoints/epoch20.checkpoint`


## Edit checkpoint
Edit dictionary of the specified checkpoint file.

*Example usage:*
`python main.py edit-checkpoint --checkpoint ./weights/stage2/test.checkpoint --key device --value cpu --typeof str`


## Benchmark
You can run benchmarking for semantic segmentation using specified weights/checkpoint on specified split of dataset.

*Example usage:*
`python main.py benchmark --weights ./weights/stage3/final.weights --dataset-split test --device gpu`

The result of the benchmark is saved in `./outputs/benchmark.txt`.


# License
>Any part of this source code should ONLY be reused for research purposes. This repo contains some modified source code from PyTorch and other sources who have been credited in source file using them.


# References
>Wang, L., Li, D., Zhu, Y., Tian, L. and Shan, Y., 2020. Dual Super-Resolution Learning for Semantic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition  (pp. 3774-3783)