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

Since our project is built on Tune-a-video, if you encounter issues with the above commands, you can also follow the steps below.

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
### <p >  Dataset download. </p>

The open-source datasets used in this paper can be accessed via the following links:

(1) CC2017: https://purr.purdue.edu/publications/2809/1

(2) HCP: https://www.humanconnectome.org/

(3) Algonauts2021: http://algonauts.csail.mit.edu/2021/index.html

```
/data
┣ 📂 CC2017_Purdue
┃   ┣ 📂 Stimuli/video_fmri_dataset/stimuli/
┃   ┃   ┣ 📜 seg1.mp4
┃   ┃   ┣ 📜 seg2.mp4
┃   ┃   ┃   ┗ ...
┃   ┃   ┣ 📜 seg18.mp4
┃   ┃   ┣ 📜 test1.mp4
┃   ┃   ┃   ┗ ...
┃   ┃   ┣ 📜 test5.mp4
┃   ┣ 📂 Subject01/video_fmri_dataset/subject1
┃   ┃   ┣ 📂 fmri
┃   ┃   ┃   ┣ 📂 seg1
┃   ┃   ┃   ┃   ┣ 📂 cifti
┃   ┃   ┃   ┃   ┃   ┣ 📜 seg1_1_Atlas.dtseries.nii
┃   ┃   ┃   ┃   ┃   ┣ 📜 seg1_2_Atlas.dtseries.nii
┃   ┃   ┃   ┃   ┣ 📂 mni
┃   ┃   ┃   ┃   ┣ 📂 raw
┃   ┃   ┃   ┣ 📂 seg2
┃   ┃   ┃   ┗ ...
┃   ┃   ┃   ┣ 📂 test5
┃   ┃   ┣ 📂 smri
┃   ┃   ┃   ┣ 📜 t1w.nii.gz
┃   ┃   ┃   ┣ 📜 t2w.nii.gz


┣ 📂 HCP
┃   ┣ 📂 Stumuli_videos
┃   ┃   ┗ 📜 clip_1.mp4
┃   ┃   ┗ 📜 clip_1.mp4
┃   ┃   ┗ ...
┃   ┃   ┗ 📜 clip_3040.mp4
┃   ┣ 📂 fMRI_response_surface
┃   ┃   ┣ 📂 100610
┃   ┃   ┃   ┣ 📜 preprocessed_fMRI.npy
┃   ┃   ┣ 📂 102816
┃   ┃   ┣ 📂 104416


┣ 📂 Algonauts2021_data
┃   ┣ 📂 AlgonautsVideos268_All_30fpsmax
┃   ┃   ┣ 📜 0001_0-0-1-6-7-2-8-0-17500167280.mp4
┃   ┃   ┣ 📜 0002_0-0-4-3146384004.mp4
┃   ┃   ┃   ┗ ...
┃   ┃   ┣ 📜 1102_meta_R-5602303_250.mp4
┃   ┣ 📂 participants_data_v2021
┃   ┃   ┣ 📂 sub01
┃   ┃   ┃   ┣ 📜 EBA.pkl
┃   ┃   ┃   ┣ 📜 FFA.pkl
┃   ┃   ┃   ┗ ...
┃   ┃   ┃   ┣ 📜 V4.pkl
┃   ┃   ┣ 📂 sub02
┃   ┃   ┃   ┗ ...
┃   ┃   ┣ 📂 sub10
```

### <p > Data preparation. </p> 
Run the code in Data_preprocess step by step to preprocess the dataset.

### <p >  Download model weights. </p>
We provide the checkpoints required to reproduce this paper in the link below. Due to the large size of the CMG model weights, we only offer the model weights trained on data from three subjects in the CC2017 dataset.  

**Additionally, if you only need to compare our model with others on new metrics, we have also included all reconstruction results on the CC2017 dataset in this folder.**

![](https://github.com/ReedOnePeck/Mind-Animator/blob/main/images/bfa84fe63a9fac6a3827d87118e2972.png)<br>

```
┣ 📂 Mind_Animator_data
┃   ┣ 📂 Testset_of_Preprocessed_datasets
┃   ┃   ┣ 📂 CC2017
┃   ┃   ┃   ┣ 📂 stimuli_clips/Test
┃   ┃   ┃   ┣ 📂 fMRI_data
┃   ┃   ┃   ┃   ┣ 📂 sub1
┃   ┃   ┃   ┃   ┃   ┣ 📂 activated_mask
┃   ┃   ┃   ┃   ┃   ┃   ┣ 📜 mask_correct.npy
┃   ┃   ┃   ┃   ┃   ┣ 📂 Test
┃   ┃   ┃   ┃   ┃   ┃   ┣ 📜 masked4500_test_data.npy
┃   ┃   ┃   ┃   ┣ 📂 sub2
┃   ┃   ┃   ┃   ┣ 📂 sub3

┃   ┣ 📂 Model_checkpoints
┃   ┃   ┣ 📂 Stable_diffusion_ckpt
┃   ┃   ┃   ┣ 📂 unet
┃   ┃   ┃   ┣ 📂 text_encoder
┃   ┃   ┃   ┣ 📂 vae
┃   ┃   ┃   ┣ 📂 tokenizer
┃   ┃   ┃   ┣ 📂 scheduler
┃   ┃   ┣ 📂 Retrieval_task
┃   ┃   ┣ 📂 Reconstruction_task

┃   ┣ 📂 Reconstruction_results
┃   ┃   ┣ 📂 CC2017
┃   ┃   ┃   ┣ 📜 reconstruction_results_sub1.zip
┃   ┃   ┃   ┣ 📜 reconstruction_results_sub2.zip
┃   ┃   ┃   ┣ 📜 reconstruction_results_sub3.zip

```

## <p align="center">  Feature extraction  </p> 
Adjust the file paths accordingly, and then run the following code in the Feature_extraction folder.
```
python Feature_extraction/semantic_feature extraction.py

python Feature_extraction/contrastive_target_extraction.py

python Feature_extraction/content_feature_extraction.py
```






