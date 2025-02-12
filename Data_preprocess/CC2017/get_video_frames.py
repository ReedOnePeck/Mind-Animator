from moviepy.editor import VideoFileClip
from PIL import Image
import os
from tqdm import tqdm

def get_frames(video_path,target_path,width,height):
    clip = VideoFileClip(video_path, target_resolution=(width,height), resize_algorithm='bilinear')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for i, frame in enumerate(clip.iter_frames()):
        if i % 8 == 0:
            im = Image.fromarray(frame)
            im.save(target_path + "/%07d.jpg" % i)
    return

#Train-----------------------------------------------------------------------------------------------------------------------

Train_videopath_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_clips/Train/'
Train_target_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Train/'
for i in tqdm(range(18)):
    for j in tqdm(range(238,240)):
        Train_videopath = Train_videopath_root + 'seg{}_{}.mp4'.format(i+1,j+1)
        Train_targetpath = Train_target_root + 'seg{}_{}'.format(i+1,j+1)
        a = get_frames(Train_videopath,Train_targetpath,512,512)

print('Trainingset Done')

#Test-----------------------------------------------------------------------------------------------------------------------
Test_videopath_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_clips/Test/'
Test_target_root = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_frames/Test/'
for i in tqdm(range(5)):
    for j in tqdm(range(238,240)):
        Test_videopath = Test_videopath_root + 'test{}_{}.mp4'.format(i + 1, j + 1)
        Test_targetpath = Test_target_root + 'test{}_{}'.format(i + 1, j + 1)
        a = get_frames(Test_videopath, Test_targetpath, 512, 512)

print('Testingset Done')



