from moviepy.video.io.VideoFileClip import VideoFileClip
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


source_file_root = '/nfs/nica-datashop/CC2017_Purdue/Stimuli/video_fmri_dataset/stimuli/'
target_file_root_Train = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_clips/Train/'
target_file_root_Test = '/nfs/diskstation/DataStation/public_dataset/CC2017_for_video_reconstruction/stimuli_clips/Test/'

#Train_video
for i in range(18):
    source_file = source_file_root + 'seg{}.mp4'.format(i + 1)
    for j in range(240):
        target_file = target_file_root_Train + 'seg{}_{}.mp4'.format(i+1,j+1)
        start_time = 2*j
        stop_time = 2*(j+1)
        a = clip_video(source_file, target_file, start_time, stop_time)
        print('')

print("Training set Done")
#Test_video
for i in range(5):
    source_file = source_file_root + 'test{}.mp4'.format(i + 1)
    for j in range(240):
        target_file = target_file_root_Test + 'test{}_{}.mp4'.format(i+1,j+1)
        start_time = 2*j
        stop_time = 2*(j+1)
        a = clip_video(source_file, target_file, start_time, stop_time)

print("Testing set Done")