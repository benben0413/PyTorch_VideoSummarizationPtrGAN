# PyTorch_VideoSummarizationPtrGAN
An implementation of Attend, Cut, and Judge

<img src='https://i.imgur.com/OuRdP91.png' width='80%' />

## Overview
We consider video summarization as a seq-seq problem, where input is image sequence and output is summariation. We use a Ptr-Net as generate to summarize input video. Different from paper applying 3D convolutional classifier as discirminator, here we **first extract visual features and concatenate them**, followed by **1D convolution classification**. The result is very similar but easier to train. This repo is for Youtube dataset and **Keyframe selection** summarization type.

## Requirements
+ Python3
+ PyTorch
+ OpenCV
+ tqdm

```bash
pip install -U torch torchvision opencv-pyhton tqdm
```

## Usage

### IPynb
+ [Preprocessing.ipynb](): Extract visual feature (**2048** dimensions) from ResNet-101
+ [Main.ipynb](): Include whole proess of model, traing, and testing

### Source
+ [Src/main.py](): Run for train/test
+ [Src/model.py](): Return generator and discriminator models
+ [Src/train.py](): Train generator (**teacher forcing**, **policy gradient**) and discriminator (**binary classification**)
+ [Src/test.py](): Evaluate generator by **F1** score
+ [Src/tools.py](): Some useful functions

## Resources
+ [Here](https://goo.gl/xyzFuL) contains preprocessed Youtube dataset and pre-trained models.

## Acknowledgement
+ We use datasets from [kezhang](https://github.com/kezhang-cs/Video-Summarization-with-LSTM) (TVSum, SumMe, Youtube) and [chengyangfu](https://github.com/chengyangfu/Pytorch-Twitch-LOL) (LoL)
+ We modified seqGAN from [suragnair](https://github.com/suragnair/seqGAN)
+ Thanks [Umbo Computer Vision](https://umbocv.ai) for collaboration