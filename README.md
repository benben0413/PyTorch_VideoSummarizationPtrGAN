# PyTorch_VideoSummarizationPtrGAN
An implementation of Attend, Cut, and Judge

<img src='https://i.imgur.com/OuRdP91.png' width='80%' />

## Overview

## Requirements
+ PyTorch
+ OpenCV
+ tqdm

```bash
	pip install -U torch torchvision opencv-pyhton tqdm
```

## Usage
+ [Preprocessing.ipynb](): Extract visual feature (**2048** dimensions) from ResNet-101
+ [Main.ipynb](): Include whole proess of model, traing, and testing
+ [Src/model.py](): Include generator and discriminator models
+ [Src/train.py](): Train generator (**teacher forcing**, **policy gradient**) and discriminator (**binary classification**)
+ [Src/test.py](): Evaluate generator by **F1** score

## Resources
+ [Here](https://goo.gl/xyzFuL) contains preprocessed Youtube dataset and pre-trained models.

## Acknowledgement
+ Thanks for datasets from [kezhang](https://github.com/kezhang-cs/Video-Summarization-with-LSTM) (TVSum, SumMe, Youtube) and [chengyangfu](https://github.com/chengyangfu/Pytorch-Twitch-LOL) (LoL)
+ Thanks [Umbo Computer Vision](https://umbocv.ai) for collaboration