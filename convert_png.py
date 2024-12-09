import openslide
import numpy as np
import PIL.Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

file_dir = '/home/yx/data/WSI'
for f_d in os.listdir(file_dir):
    for file in os.listdir(os.path.join(file_dir, f_d)):
        try:
            test = openslide.open_slide(os.path.join(file_dir, f_d, file))
            print(file)
            img = np.array(test.read_region((0, 0), 1, test.level_dimensions[1]))
            output_path = './WSI'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            png_name = file.split('.')[0]
            PIL.Image.fromarray(img).save(os.path.join(output_path, png_name + '.png'))
        except Exception as e:
            pass