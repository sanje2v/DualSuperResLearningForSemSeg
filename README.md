# Implementation of 'Dual Super Resolution Learning For Semantic Segmentation', CVPR 2020 paper
[This paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.html) combines Super-Resolution and Feature Affinity learning to improve traditional semantic segmentation model.


# Getting started

Before training or benchmarking, please download 'gtFine.zip' and 'leftImg8bit.zip' for [**Cityscapes dataset**](https://www.cityscapes-dataset.com/) and unzip them under './datasets/Cityscapes/data'.

All commands are invoked via `main.py` script. For instance to train, you would use something like `python main.py --train [...]`. Use `python main.py --help` to view all supported actions.

# Commands
## Training
Perform training on the '*train*' split of **Cityscapes dataset**.

*Example usage:*
`python main.py train --stage 1 --description "Stage 1 training" --epochs 200 --batch_size 6 --device gpu --checkpoint_history 40 --num_workers 4 --val_interval 5` 

## Resume training
Resume training from an interrupted session using checkpoint file (that have .checkpoints extension and are autosaved under './weights/checkpoints').

`python main.py resume_train --checkpoint ./weights/checkpoints/epoch50.checkpoint`

## Testing
Perform inference on a specified image file with specified weights.

*Example usage:*
`python main.py test --image_file ~/Pictures/input.jpg --weights ./weights/stage3/final.weights --device gpu`

The result of the inference is saved in `./outputs/<image_filename>.png`.

## Weights pruning
Weights trained in stage 2 and 3 will have weights for parts of model not needed for inference. So, weights pruning will remove all these weights.

*Example usage:*
`python main.py prune_weights --src_weights ./weights/stage3/autosaves/epoch50.checkpoint --dest_weights ./output/inference.weights`

## Benchmark
You can run benchmarking for semantic segmentation using specified weights/checkpoint on specified split of dataset.

*Example usage:*
`python main.py benchmark --weights ./weights/stage3/final.weights --dataset_split test --device gpu`

The result of the benchmark is saved in `./outputs/benchmark.txt`.

# License

Any part of this source code should ONLY be reused for research purposes. This repo contains some modified source code from PyTorch.

# References
>Wang, L., Li, D., Zhu, Y., Tian, L. and Shan, Y., 2020. Dual Super-Resolution Learning for Semantic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition  (pp. 3774-3783)