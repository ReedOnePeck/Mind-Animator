from moviepy.editor import VideoFileClip
from PIL import Image
import os
from tqdm import tqdm

def get_frames(video_path,target_path,width,height):
    clip = VideoFileClip(video_path, target_resolution=(width,height), resize_algorithm='bilinear')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for i, frame in enumerate(clip.iter_frames()):
        if i % 3 == 0:
            im = Image.fromarray(frame)
            im.save(target_path + "/%07d.jpg" % i)
    return

videopath_root = '/data0/home/luyizhuo/HCP预处理/Stumuli_videos/'
frames_save_root = '/data0/home/luyizhuo/HCP预处理/video_frames/'
video_path = videopath_root + 'clip_{}.mp4'.format(0+1)
frame_path = frames_save_root + 'video{}_frames'.format(0+1)
get_frames(video_path, frame_path, 512, 512)

"""
videopath_root = '/data0/home/luyizhuo/HCP预处理/Stumuli_videos/'
frames_save_root = '/data0/home/luyizhuo/HCP预处理/video_frames/'

for idx in tqdm(range(3040)):
    video_path = videopath_root + 'clip_{}.mp4'.format(idx+1)
    frame_path = frames_save_root + 'video{}_frames'.format(idx+1)
    get_frames(video_path, frame_path, 512, 512)
"""

