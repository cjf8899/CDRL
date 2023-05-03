# Unsupervised Change Detection Based on Image Reconstruction Loss

## release note

23.05.03 - We have published an extended version of this paper, [CDRL-SA](https://github.com/cjf8899/CDRL-SA).
22.07.07 - Add : evaluate metric (note that all evaluate metrics is mean value. (0, 1))

> [**Unsupervised Change Detection Based on Image Reconstruction Loss**](https://arxiv.org/abs/2204.01200)
> 
> Hyeoncheol Noh, Jingi Ju, Minseok Seo, Jongchan Park, Dong-Geol Choi
> 
> *[arXiv 2204.01200](https://arxiv.org/abs/2204.01200)*
> 
> *[Project Page](https://jujingi.github.io/cdrl/)*

## Results

Result videos are the results of the diffence map for each threshold. We used a threshold of 0.7.

<img src="https://user-images.githubusercontent.com/53032349/163550463-1d0467aa-eff8-4e6e-9c57-2bf6fe493a4f.gif" alt="Change Pair Result" width="400"/>   <img src="https://user-images.githubusercontent.com/53032349/163550507-38d948c0-6c0d-4b9d-b7cd-5472aa97fd50.gif" alt="Unchange Pair Result" width="400"/>

## Abstract
To train the change detector, bi-temporal images taken at different times in the same area are used. However, collecting labeled bi-temporal images is expensive and time consuming. To solve this problem, various unsupervised change detection methods have been proposed, but they still require unlabeled bi-temporal images. In this paper, we propose unsupervised change detection based on image reconstruction loss using only unlabeled single temporal single image. The image reconstruction model is trained to reconstruct the original source image by receiving the source image and the photometrically transformed source image as a pair. During inference, the model receives bitemporal images as the input, and tries to reconstruct one of the inputs. The changed region between bi-temporal images shows high reconstruction loss. Our change detector showed significant performance in various change detection benchmark datasets even though only a single temporal single source image was used. The code and trained models will be publicly available for reproducibility.
<p align="center"><img src="https://user-images.githubusercontent.com/53032349/163549501-817b7852-38c2-45d7-b287-f9b65f51c9c2.png" width="90%" height="90%" title="70px" alt="memoryblock"></p>


## Installation

### Step1. Install CDRL.
```shell
git clone https://github.com/cjf8899/CDRL.git
cd CDRL
pip install -r requirements.txt
```

### Step2. Creating a Pseudo-Unchange Image.

Download [LEVIR-CD](https://justchenhao.github.io/LEVIR/), [LEVIR-CD_A2B_B2A](https://drive.google.com/file/d/1-LERpM7GOxviKna47bbO_mLQON3Q0YcA/view?usp=sharing) and put them under <CDRL_HOME>/datasets in the following structure:

```
CDRL/datasets
       |——————LEVIR-CD
       |        └——————train
       |        └——————val
       |        └——————test
       └——————LEVIR-CD_A2B_B2A
                └——————train
                └——————val

```

The Photometric Transform model we used the CycleGAN code of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We provide the model's weights.

* latest_net_G_A [[google]](https://drive.google.com/file/d/1M7fIJo6koqLFqXVjKG0PWHRWlTPN5BZV/view?usp=sharing)
* latest_net_G_B [[google]](https://drive.google.com/file/d/1k_tGVaI-4_Wn6-eLT0qvm8YsIz9oDqnS/view?usp=sharing)

<img src="https://user-images.githubusercontent.com/53032349/163725551-d0a229e1-7568-4040-90f2-093de0e72452.png" width="400"/>   <img src="https://user-images.githubusercontent.com/53032349/163725563-d57961bd-ead9-4edd-89fe-13a4b912276e.png" width="400"/>


## Training
```shell
python main.py --root_path ./datasets/ --dataset_name LEVIR-CD --save_name levir 
```

## Creating a Difference map.
```shell
python test.py --root_path ./datasets/ --dataset_name LEVIR-CD --save_name levir --save_visual
```

