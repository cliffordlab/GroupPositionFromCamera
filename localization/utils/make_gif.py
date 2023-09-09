import os
from moviepy.editor import ImageClip, concatenate_videoclips, ImageSequenceClip, VideoFileClip
from pprint import pprint

dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid\group_th_43.5'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid\kps_smth\pi136.pi.bmi.emory.edu'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid\kps_smth\pi148.pi.bmi.emory.edu'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid\frames\pi136.pi.bmi.emory.edu'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid\frames\pi148.pi.bmi.emory.edu'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid_ori\orientation\pi148.pi.bmi.emory.edu\000\occup_ori_ep6'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid_ori\orientation\pi148.pi.bmi.emory.edu\000\kps_track'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid_ori\orientation\pi136.pi.bmi.emory.edu\000\kps_track'
dir_data = r'D:\Research\Emory\EP6\exps\video\2022.03.18_16.13_grid_ori\orientation\pi136.pi.bmi.emory.edu\000\occup_ori_ep6'

list_imgs = os.listdir(dir_data)
list_imgs.sort()
list_imgs = [dir_data + f'/{item}' for item in list_imgs if '.gif' not in item]
pprint (list_imgs)
# assert False

file_gif = dir_data + '/sequence.gif'

clip = ImageSequenceClip(list_imgs, fps=5)
clip.write_gif(file_gif)
print ('save in ...', file_gif)
