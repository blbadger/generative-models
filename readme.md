## README

### Generative Models Introduction

This repository contains a number of generative deep learning models, training pipelines, and analysis tools.  The code is currently used for research and is not optimized for production nor is it particularly clean as a result.

Trained model weights may be found [here](https://drive.google.com/drive/folders/1iqJRqxTuJpYGZKjcnBBXzvFpSCLmGfbi?usp=sharing).

### autoencoders

Contains experimental work on the use of overcomplete autoencoders for input synthesis and denoising.

### diffusion

Denoising Diffusion Inversion models, training, and inference protocols.  The Unet-based models and q and p calculations are forked from the official Pytorch implementation by Phil Wang found [here](https://github.com/lucidrains/denoising-diffusion-pytorch), which implements a model introduced by [Ho and colleages](https://arxiv.org/abs/2006.11239). 

### gans

Generative Adversarial Networks

### trainable_priors

Autoencoders trained to provide priors on input representation using gradient matching.  Not particularly successful at present.
