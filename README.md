# <p align="center"> Mind-Animator </p> 
This is the official code for the paper "Animate Your Thoughts: Reconstruction of Dynamic Natural Vision from Human Brain Activity"[**ICLR 2025**] [Project page](https://mind-animator-design.github.io/)

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

## <p align="center">  Feature decoding  </p> 

All hyperparameters have been set in the code according to the values reported in the paper. You only need to adjust the file paths accordingly, and then run the following code in the Feature_decoding folder, taking Subject 1 as an example:

```
python Feature_decoding/train_semantic_decoder.py --model_dir your_model_save_path --figure_dir your_figure_save_path --subj_ID 1

python Feature_decoding/train_structure_decoder.py --model_dir your_model_save_path --figure_dir your_figure_save_path --subj_ID 1

python Feature_decoding/train_CMG.py --model_dir your_model_save_path --figure_dir your_figure_save_path --subj_ID 1
```
**The above patch_size setting for CMG is set to 64 by default, which can be very resource-intensive and carries a risk of overfitting. If you're interested, you can set a smaller patch_size to train the model in models/CMG_model_with_more_patchsize.py. However, based on our experimental results, we found that when the patch_size is set smaller, the model is more prone to overfitting, and the reconstructed videos tend to become 'grid-like' (you can also validate this by using the checkpoints we provided for Subject 1 in the CC2017 dataset).**


## <p align="center">  Video reconstruction  </p> 
```
python Reconstruction/video_recons.py --video_save_folder1 your_model_save_recon --video_save_folder1 your_model_save_recon_and_gt --subj_ID 1 --random_seed 42
```

## <p align="center">  Retrieval  </p> 
This project has been ongoing for a long time, and I can no longer find the training code. If you're interested, you can use the checkpoints we provided to run Retrieval/small_set.py and Retrieval/Large_set.py.
```
python Retrieval/small_set.py
python Retrieval/Large_set.py

```

## <p align="center">  Evaluation metric calculation  </p> 
run the relevant code in the Evaluation_Metrics folder. Note that the calculation of the VIFI_CLIP metric requires the environment from [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP), and you will need to convert the .mp4 files to .avi format in advance.




## <p align="center">  Explainable_analysis  </p> 

*  Note that what we aim to test with the shuffle test is whether the correctly decoded videos contain the correct motion features. Therefore, the experiment videos for the shuffle test are only those videos with correct semantic decoding.
*  By following my tutorial and running the code in Explainable_analysis/Cortical_visualization step by step, you can obtain the cortical surface projection maps displayed in the paper.

## <p align="center">  Acknowledgments  </p> 

We would like to express our gratitude to Prof.Jack L. Gallant and Prof.Shinji Nishimoto for their
pioneering exploration in the field of video reconstruction and for their high-quality code. We are
grateful to Prof.Juan Helen Zhou and Dr.Zijiao Chen for their patient answers to our questions and
for making all the results of the Mind-video test set public. We also extend our thanks to Prof.Michal
Irani, Dr.Ganit Kupershmidt, and Dr.Roman Beliy for providing us with all the reconstruction results
of their models on the test set.

We would like to express our appreciation to Prof.Zhongming Liu and Dr.Haiguang Wen for their
open-sourced high-quality video-fMRI dataset and the preprocessing procedures. Our gratitude
also goes to the Human Connectome Project (HCP) for providing a large-scale fMRI dataset and
cortical visualization tools. We are thankful to the Algonauts2021 competition for providing a set of
pre-processed video-fMRI data from multiple subjects.

We also appreciate the Tune-a-video team for their open-source video editing framework, which allows
us to reconstruct videos without introducing additional motion information.



## <p align="center">  Cite  </p> 
Please cite our paper if you use this code in your own work:
```
@article{lu2024animate,
  title={Animate Your Thoughts: Decoupled Reconstruction of Dynamic Natural Vision from Slow Brain Activity},
  author={Lu, Yizhuo and Du, Changde and Wang, Chong and Zhu, Xuanliu and Jiang, Liuyun and He, Huiguang},
  journal={arXiv preprint arXiv:2405.03280},
  year={2024}
}

```




