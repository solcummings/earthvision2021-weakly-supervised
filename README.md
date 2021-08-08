# Siamese Attention U-Net for Multi-Class Change Detection
1st place solution for the EARTHVISION 2021 DynamicEarthNet Challenge - Weakly-Supervised Multi-Class Change Detection Track at CVPRW 2021  
[[Challenge Site]](https://competitions.codalab.org/competitions/30441) 
[[Presentation Slides]](./examples/earthvision2021_presentation.pdf) 

## Overview
This work introduces a pixel-wise change detection network named Siamese Attention U-Net that incorporates attention mechanisms in the Siamese U-Net architecture. Experiments show the architectural change alongside training strategies such as semi-supervised learning produce more robust models.  
  
![Siamese Attention U-Net](./examples/siamese_attention_unet.png)

### Results
1. Attention block  

![Proposed Attention Block](./examples/attention_block_proposed.png)

|Attention block location|mean IoU (val)|
|:-:|:-:|
|None|0.2635|
|Skip connection|0.2603|
|Up-sample|0.2658|

2. Loss function  

|Loss function|mean IoU (val)|
|:-:|:-:|
|Jaccard|0.2658|
|Dice|0.2668|
|Ensemble|0.2676|

3. Semi-supervised learning  

|Pseudo labels|Loss function|mean IoU (val)|mean IoU (test)|
|:-:|:-:|:-:|:-:|
|None|Jaccard|0.2658||
|val+test|Jaccard|0.2669||
|None|Dice|0.2668|
|val+test|Dice|0.2674||
|None|Ensemble|0.2676||
|val+test|Ensemble|0.2684|0.2423|


## Usage
### Dependencies
- gdal
- numpy
- pandas
- pillow
- pytorch
- pyyaml
- torchvision
- tqdm

### Downloading and Preprocessing Data

### Training

### Predicting


