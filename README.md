# Unsupervised Change Detection Based on Image Reconstruction Loss

> [**Unsupervised Change Detection Based on Image Reconstruction Loss**](https://arxiv.org/abs/2204.01200)
> 
> Hyeoncheol Noh, Jingi Ju, Minseok Seo, Jongchan Park, Dong-Geol Choi
> 
> *[arXiv 2204.01200](https://arxiv.org/abs/2204.01200)*
> *[Project Page](https://jujingi.github.io/cdrl/)*

## Result

<img src="https://user-images.githubusercontent.com/53032349/163550463-1d0467aa-eff8-4e6e-9c57-2bf6fe493a4f.gif" width="400"/>   <img src="https://user-images.githubusercontent.com/53032349/163550507-38d948c0-6c0d-4b9d-b7cd-5472aa97fd50.gif" width="400"/>

## Abstract
To train the change detector, bi-temporal images taken at different times in the same area are used. However, collecting labeled bi-temporal images is expensive and time consuming. To solve this problem, various unsupervised change detection methods have been proposed, but they still require unlabeled bi-temporal images. In this paper, we propose unsupervised change detection based on image reconstruction loss using only unlabeled single temporal single image. The image reconstruction model is trained to reconstruct the original source image by receiving the source image and the photometrically transformed source image as a pair. During inference, the model receives bitemporal images as the input, and tries to reconstruct one of the inputs. The changed region between bi-temporal images shows high reconstruction loss. Our change detector showed significant performance in various change detection benchmark datasets even though only a single temporal single source image was used. The code and trained models will be publicly available for reproducibility.
<p align="center"><img src="https://user-images.githubusercontent.com/53032349/163549501-817b7852-38c2-45d7-b287-f9b65f51c9c2.png" width="70%" height="70%" title="70px" alt="memoryblock"></p>


## Installation
TODO
