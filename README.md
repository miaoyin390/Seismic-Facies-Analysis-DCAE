# Unsupervised seismic facies analysis via deep convolutional autoencoders
This repository is an implement of my paper, which is as follows:
> Qian F, Yin M, Liu X Y, et al. Unsupervised seismic facies analysis via deep convolutional autoencoders[J]. Geophysics, 2018, 83(3): A39-A43. https://doi.org/10.1190/geo2017-0524.1

## Abstract
One of the most important goals of seismic stratigraphy studies is to interpret the elements of the seismic facies with respect to the geologic environment. Prestack seismic data carry rich information that can help us get higher resolution and more accurate facies maps. Therefore, it is promising to use prestack seismic data for the seismic facies recognition task. However, because each identified object changes from the poststack trace vectors to a prestack trace matrix, effective feature extraction becomes more challenging. We have developed a novel data-driven offset-temporal feature extraction approach using the deep convolutional autoencoder (DCAE). As an unsupervised deep learning method, DCAE learns nonlinear, discriminant, and invariant features from unlabeled data. Then, seismic facies analysis can be accomplished through the use of conventional classification or clustering techniques (e.g., K-means or self-organizing maps). Using a physical model and field prestack seismic surveys, we comprehensively determine the effectiveness of our scheme. Our results indicate that DCAE provides a much higher resolution than the conventional methods and offers the potential to significantly highlight stratigraphic and depositional information.

## Experiments
![result1](https://github.com/ymthink/Seismic-Facies-Analysis-DCAE/blob/master/result1.png)
Figure 1. Facies maps of physical model. (a) The preparation of the physical model, (b) the prototype of the physical model, (c) the result using DCAE based on prestack data, (d) the result using WT-SOM based on poststack data, and (e) the result using WTMM-PCA-SOM based on prestack data.

![result2](https://github.com/ymthink/Seismic-Facies-Analysis-DCAE/blob/master/result2.png)
Figure 2. Facies maps of the target horizon in the Liziba survey. (a) Coherence image based on poststack data, (b) the result using DCAE based on prestack data, (c) the result using WTMM-SOM based on poststack data, and (d) the result using WTMM-PCA-SOM based on prestack data.

## Citation
```
@article{qian2018unsupervised,
  title={Unsupervised seismic facies analysis via deep convolutional autoencoders},
  author={Qian, Feng and Yin, Miao and Liu, Xiao-Yang and Wang, Yao-Jun and Lu, Cai and Hu, Guang-Min},
  journal={Geophysics},
  volume={83},
  number={3},
  pages={A39--A43},
  year={2018},
  publisher={Society of Exploration Geophysicists}
}
```
