from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import itertools
def clip_video(source_file, target_file, start_time, stop_time):
    """
    :param source_file: 原视频的路径，mp4格式
    :param target_file: 生成的目标视频路径，mp4格式
    :param start_time: 剪切的起始时间点(第start_time秒)
    :param stop_time: 剪切的结束时间点(第stop_time秒)
    :return:
    """
    source_video = VideoFileClip(source_file)
    video = source_video.subclip(int(start_time), int(stop_time))  # 执行剪切操作
    video.write_videofile(target_file)  # 输出文件
    return

clips = np.load('/data0/home/luyizhuo/HCP预处理/clip_times_24.npy', allow_pickle=True).item()
video_names = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']
video_path = '/nfs/nica-datashop/HCP_7T_Movie/movie_stimulus/Post_20140821_version/'
save_path = '/data0/home/luyizhuo/HCP预处理/Stumuli_videos/clip_{}.mp4'

"""
video_file = video_path + '7T_MOVIE1_CC1_v2.mp4'
solid_clips = clips['1']
start_times = []
for c in range(len(solid_clips)):
    start_times.append(np.arange(solid_clips[c, 0] /24, solid_clips[c, 1]/24 ).astype('int'))
start_times = list(itertools.chain(*start_times))
for j, idx in enumerate(start_times):
    clip_video(video_file, save_path.format(j+1), idx, idx+1)
"""

#728,792,745,775

for jj, v_name in enumerate(video_names):
    if jj+1 == 1:
        valid = 1
    if jj + 1 == 2:
        valid = 728+1
    if jj+1 == 3:
        valid = 728+792+1
    if jj+1 == 4:
        valid = 728+792+745+1

    video_file = video_path + v_name
    solid_clips = clips['{}'.format( jj+1 )]
    start_times = []
    for c in range(len(solid_clips)):
        start_times.append(np.arange(solid_clips[c, 0] / 24, solid_clips[c, 1] / 24).astype('int'))

    start_times = list(itertools.chain(*start_times))
    for j, idx in enumerate(start_times):
        clip_video(video_file, save_path.format(j + valid), idx, idx + 1)













