# <p align="center"> Mind-Animator </p> 
This is the official code for the paper "Animate Your Thoughts: Reconstruction of Dynamic Natural Vision from Human Brain Activity"[**ICLR 2025**] [Project page](https://dl.acm.org/doi/10.1145/3581783.3613832)

## <p align="center">  Related works  </p> 
![](https://github.com/ReedOnePeck/Mind-Animator/blob/main/images/related_works.png)<br>

## <p align="center">  Schematic diagram of Mind-Animator  </p> 
![](https://github.com/ReedOnePeck/Mind-Animator/blob/main/images/method.png)<br>


## <p align="center">  News  </p> 
- Jan. 25, 2025. Project release.
- Jan. 23, 2025. Our paper is accpeted at ICLR2025!

# <p align="center"> Steps to reproduce Mind-Animator </p> 
## <p align="center">  Preliminaries  </p> 
This code was developed and tested with:

*  Python version 3.9.16
*  PyTorch version 1.12.1
*  A100 80G

## <p align="center">  Environment setup  </p> 
Create and activate conda environment named ```Mind-animator``` from our ```environment_MA.yml```
```sh
conda env create -f environment_MA.yml
conda activate Mind-animator
```

Since our project is built on Tune-a-video, if you encounter issues with the above commands, you can also follow the steps below one by one.

*  Create a virtual environment for Tune-a-video.
```sh
pip install -r Tune-a-video-requirements.txt
```
*  Install CLIP.
```sh
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
*  Install the remaining packages as needed.

## <p align="center">  Data preparation  </p> 
*  Dataset download.
The open-source datasets used in this paper can be accessed via the following links:

(1) CC2017: https://purr.purdue.edu/publications/2809/1

(2) HCP: https://www.humanconnectome.org/

(3) Algonauts2021: http://algonauts.csail.mit.edu/2021/index.html





