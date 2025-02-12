from moviepy.editor import VideoFileClip
from PIL import Image
import os
import re
from tqdm import tqdm

def get_frames(video_path,target_path,width,height):
    clip = VideoFileClip(video_path, target_resolution=(width,height), resize_algorithm='bilinear')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for i, frame in enumerate(clip.iter_frames()):
        if i <=7:
            im = Image.fromarray(frame).resize((512,512))
            im.save(target_path + "/%02d.jpg" % i)
    return

videopath_root = '/nfs/nica-datashop/Algonauts2021_data/AlgonautsVideos268_All_30fpsmax/'
frames_save_root = '/data0/home/luyizhuo/Algonauts2021_paired_data/Stimuli/'

names = os.listdir(videopath_root)
number_pattern = re.compile(r'(\d+)_')
def extract_number(filename):
    match = number_pattern.search(filename)
    if match:
        return int(match.group(1))
    return float('inf')
sorted_file_list = sorted(names, key=extract_number)

sorted_names = []
for filename in sorted_file_list:
    sorted_names.append(filename)

for idx in tqdm(range(832,1000)):
    video_path = videopath_root + sorted_names[idx]
    frame_path = frames_save_root + 'video_{}/'.format(idx + 1)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    get_frames(video_path, frame_path, 256, 256)


