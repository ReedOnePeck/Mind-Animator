from PIL import Image
import os
from tqdm import tqdm


def process_image(image_path, output_path):
    # 打开图片
    img = Image.open(image_path)

    # 原始图片尺寸
    original_size = img.size

    # 计算裁剪尺寸
    left = (original_size[0] - 420) // 2
    top = (original_size[1] - 420) // 2
    right = (original_size[0] + 420) // 2
    bottom = (original_size[1] + 420) // 2
    center_crop_area = (left, top, right, bottom)

    # 中心裁剪
    img_cropped = img.crop(center_crop_area)

    # 调整大小到512x512
    img_resized = img_cropped.resize((512, 512))

    # 保存处理后的图片
    img_resized.save(output_path)

old_folder_root = '/data0/home/luyizhuo/HCP预处理/video_frames/'
new_folder_root = '/data0/home/luyizhuo/HCP预处理/video_frames_new/'

for i in tqdm(range(1,3041)):
    old_folder_path = old_folder_root + 'video{}_frames/'.format(i)
    new_folder_path = new_folder_root + 'video{}_frames/'.format(i)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    for img_name in ['0000000.jpg','0000003.jpg','0000006.jpg','0000009.jpg','0000012.jpg','0000015.jpg','0000018.jpg','0000021.jpg']:
        old_img_path = old_folder_path + img_name
        new_img_path = new_folder_path + img_name
        process_image(old_img_path, new_img_path)



